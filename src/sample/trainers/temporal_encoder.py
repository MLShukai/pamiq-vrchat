from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import cast, override

import mlflow
import torch
from pamiq_core import DataUser
from pamiq_core.torch import OptimizersSetup, TorchTrainer, get_device
from tensordict import TensorDict
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from sample.data import BufferName, DataKey
from sample.models import ModelName
from sample.models.temporal_encoder import TemporalEncoder

from .sampler import RandomTimeSeriesSampler

OPTIMIZER_NAME = "optimizer"


type TensorOrTensorDict = torch.Tensor | TensorDict


def transpose_and_stack_collator(
    batch_items: Iterable[tuple[TensorOrTensorDict, ...]],
) -> tuple[TensorOrTensorDict, ...]:
    """Transposes a list of tuples and stacks each resulting column into a
    single tensor.

    Example:
        >>> tensors = [
        ...     (tensor([1]), tensor([2])),
        ...     (tensor([3]), tensor([4])),
        ... ]
        >>> transpose_and_stack(tensors)
        (tensor([1, 3]), tensor([2, 4]))

    Args:
        batch_items: List of tuples containing tensors or tensor dicts to be processed.
            All tuples must have the same length.

    Returns:
        Tuple of stacked tensors or tensor dicts, where each element corresponds
        to a column in the original data.
    """
    return tuple(torch.stack(item) for item in zip(*batch_items))


class TemporalEncoderTrainer(TorchTrainer):
    """Trainer for Temporal Encoder that handles sequential data with multiple
    modalities."""

    @override
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[Tensor]],
        partial_sampler: partial[RandomTimeSeriesSampler],
        partial_optimizer: partial[Optimizer],
        max_epochs: int = 1,
        data_user_name: str = BufferName.TEMPORAL,
        min_buffer_size: int = 2,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize the TemporalEncoder trainer.

        Args:
            partial_dataloader: Partially configured DataLoader to be used with
                dynamically created datasets during training.
            partial_sampler: Partially configured RandomTimeSeriesSampler for
                time series data sampling.
            partial_optimizer: Partially configured optimizer to be used with
                the model parameters.
            max_epochs: Maximum number of epochs to train per training session.
            data_user_name: Name of the data user providing training data.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
        """
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.data_user_name = data_user_name
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.partial_sampler = partial_sampler
        self.max_epochs = max_epochs
        self.global_step = 0

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the
        trainer."""
        super().on_data_users_attached()
        self.temporal_data_user: DataUser[Tensor | TensorDict] = self.get_data_user(
            self.data_user_name
        )

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer."""
        super().on_training_models_attached()
        self.temporal_encoder = self.get_torch_training_model(
            ModelName.TEMPORAL_ENCODER, TemporalEncoder
        )

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for temporal encoder training.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {
            OPTIMIZER_NAME: self.partial_optimizer(
                self.temporal_encoder.model.parameters()
            )
        }

    @override
    def train(self) -> None:
        """Execute temporal encoder training process."""
        # Get dataset from data user
        data = self.temporal_data_user.get_data()

        dataset = TensorDataset(
            # Ignore pyright error because TensorDict api is compatible with torch.Tensor api.
            torch.stack(list(data[DataKey.OBSERVATION])),  # pyright: ignore[reportArgumentType]
            torch.stack(cast(list[Tensor], list(data[DataKey.HIDDEN]))),
        )
        sampler = self.partial_sampler(dataset)
        dataloader = self.partial_dataloader(
            dataset=dataset, sampler=sampler, collate_fn=transpose_and_stack_collator
        )
        device = get_device(self.temporal_encoder.model)

        for _ in range(self.max_epochs):
            batch: tuple[TensorDict, Tensor]
            for batch in dataloader:
                batch_observations, batch_hiddens = batch

                # Move data to device
                batch_observations = batch_observations.to(device)
                batch_hiddens = batch_hiddens.to(device)

                # Prepare sequences
                observations = batch_observations[:, :-1]  # o_0:T-1
                hiddens = batch_hiddens[:, 0]  # h_0
                observations_next = batch_observations[:, 1:]  # o_1:T

                self.optimizers[OPTIMIZER_NAME].zero_grad()

                # Forward pass
                _, _, obs_hat_dists = self.temporal_encoder.model(observations, hiddens)

                # Calculate losses for each modality
                total_loss = torch.tensor(0.0, device=device)
                modality_losses = {}

                for modality, dist in obs_hat_dists.items():
                    modal_loss = -dist.log_prob(observations_next[modality]).mean()
                    modality_losses[modality] = modal_loss
                    total_loss += modal_loss

                # Backward pass
                total_loss.backward()

                self.optimizers[OPTIMIZER_NAME].step()

                # Logging
                metrics = {
                    "loss/total": total_loss.item(),
                }
                for modality, loss in modality_losses.items():
                    metrics[f"loss/{modality}"] = loss.item()

                # Log gradient norm
                if self.temporal_encoder.model.parameters():
                    grad_norm = torch.cat(
                        [
                            p.grad.flatten()
                            for p in self.temporal_encoder.model.parameters()
                            if p.grad is not None
                        ]
                    ).norm()
                    metrics["grad_norm"] = grad_norm.item()

                mlflow.log_metrics(
                    {f"temporal-encoder/{tag}": v for tag, v in metrics.items()},
                    self.global_step,
                )

                self.global_step += 1

    @override
    def save_state(self, path: Path) -> None:
        """Save trainer state to disk."""
        super().save_state(path)
        path.mkdir(exist_ok=True)
        (path / "global_step").write_text(str(self.global_step), "utf-8")

    @override
    def load_state(self, path: Path) -> None:
        """Load trainer state from disk."""
        super().load_state(path)
        self.global_step = int((path / "global_step").read_text("utf-8"))
