from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import override

import mlflow
import torch
from pamiq_core import DataUser
from pamiq_core.torch import OptimizersSetup, TorchTrainer, get_device
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from sample.data import BufferName, DataKey
from sample.models import ModelName
from sample.models.forward_dynamics import ForwardDynamics

from .sampler import RandomTimeSeriesSampler

OPTIMIZER_NAME = "optimizer"


class ImaginingForwardDynamicsTrainer(TorchTrainer):
    """Trainer for the ForwardDynamics model.

    This trainer implements the training loop for the ForwardDynamics
    model, which predicts the next observation distribution given the
    current observation and action. It uses a recurrent core model to
    maintain hidden state across sequential predictions.
    """

    def __init__(
        self,
        partial_optimizer: partial[Optimizer],
        seq_len: int = 1,
        max_samples: int = 1,
        batch_size: int = 1,
        max_epochs: int = 1,
        data_user_name: str = BufferName.FORWARD_DYNAMICS,
        imagination_length: int = 1,
        min_buffer_size: int = 0,
        min_new_data_count: int = 0,
        imagination_average_method: Callable[[Tensor], Tensor] = torch.mean,
    ) -> None:
        """Initialize the ForwardDynamicsTrainer.

        Args:
            partial_optimizer: Partially configured optimizer to be used with
                the model parameters.
            seq_len: Sequence length per batch.
            max_samples: Max number of sample from dataset in 1 epoch.
            batch_size: Data sample size of 1 batch.
            max_epochs: Maximum number of epochs to train per training session.
            data_user_name: Name of the data user providing training data.
            imagination_length: Length of the imagination sequence.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
            imagenation_average_method: Method to average the loss over the imagination sequence.
        """
        if min_buffer_size < imagination_length + seq_len:
            raise ValueError(
                "Buffer size must be greater than imagination length + sequence length."
            )
        if imagination_length < 1:
            raise ValueError("Imagination length must be greater than 0")

        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.partial_optimizer = partial_optimizer
        self.partial_sampler = partial(
            RandomTimeSeriesSampler,
            sequence_length=seq_len + imagination_length,
            max_samples=max_samples,
        )
        self.partial_dataloader = partial(DataLoader, batch_size=batch_size)
        self.max_epochs = max_epochs
        self.data_user_name = data_user_name
        self.imagination_length = imagination_length

        self.imagination_average_method = imagination_average_method
        self.global_step = 0

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        data users are attached to the trainer. It retrieves and stores
        references to the required data users for convenient access
        during training.
        """
        super().on_data_users_attached()
        self.forward_dynamics_data_user: DataUser[Tensor] = self.get_data_user(
            self.data_user_name
        )

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        training models are attached to the trainer. It retrieves and
        stores references to the ForwardDynamics model for convenient
        access during training.
        """

        super().on_training_models_attached()
        self.forward_dynamics = self.get_torch_training_model(
            ModelName.FORWARD_DYNAMICS, ForwardDynamics
        )

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for the ForwardDynamics model. This method is
        called automatically by the PAMIQ framework to set up optimizers for
        the training process. It uses the `partial_optimizer` function to
        create an optimizer for the ForwardDynamics model's parameters.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {
            OPTIMIZER_NAME: self.partial_optimizer(
                self.forward_dynamics.model.parameters()
            )
        }

    @override
    def train(self) -> None:
        """Execute ForwardDynamics training process.

        This method implements the core ForwardDynamics training loop:
        1. Creates a dataset and dataloader
        2. For each batch:
            - Moves data to the appropriate device
            - Splits observations, actions, and hidden states
            - Computes the next observation distribution
            - Calculates the loss
            - Backpropagates the loss
            - Updates the model parameters
        3. Logs the loss to MLflow
        4. Increments the global step counter
        """

        data = self.forward_dynamics_data_user.get_data()
        dataset = TensorDataset(
            torch.stack(list(data[DataKey.OBSERVATION])),
            torch.stack(list(data[DataKey.ACTION])),
            torch.stack(list(data[DataKey.HIDDEN])),
        )
        sampler = self.partial_sampler(dataset=dataset)
        dataloader = self.partial_dataloader(dataset=dataset, sampler=sampler)
        device = get_device(self.forward_dynamics.model)

        for _ in range(self.max_epochs):
            batch: tuple[Tensor, Tensor, Tensor]
            for batch in dataloader:
                observations, actions, hiddens = batch
                observations = observations.to(device)
                actions = actions.to(device)
                obs_imaginations, hiddens = (
                    observations[:, : -self.imagination_length],
                    hiddens[:, 0].to(device),
                )

                self.optimizers[OPTIMIZER_NAME].zero_grad()

                obses_next_hat_dist: Distribution
                loss_imaginations: list[Tensor] = []
                for i in range(self.imagination_length):
                    action_imaginations = actions[
                        :, i : -self.imagination_length + i
                    ]  # a_i:i+T-H, (B, T-H, *)
                    obs_targets = observations[
                        :,
                        i + 1 : observations.size(1) - self.imagination_length + i + 1,
                    ]  # o_i+1:T-H+i+1, (B, T-H, *)
                    if i > 0:
                        action_imaginations = action_imaginations.flatten(
                            0, 1
                        )  # (B', *)
                        obs_targets = obs_targets.flatten(0, 1)  # (B', *)

                    obses_next_hat_dist, next_hiddens = self.forward_dynamics.model(
                        obs_imaginations, action_imaginations, hiddens
                    )
                    loss = -obses_next_hat_dist.log_prob(obs_targets).mean()
                    loss_imaginations.append(loss)
                    obs_imaginations = obses_next_hat_dist.rsample()

                    if i == 0:
                        obs_imaginations = obs_imaginations.flatten(
                            0, 1
                        )  # (B, T-H, *) -> (B', *)
                        hiddens = next_hiddens.movedim(2, 1).flatten(
                            0, 1
                        )  # h'_i, (B, D, T-H, *) -> (B, T-H, D, *) -> (B', D, *)

                loss = self.imagination_average_method(torch.stack(loss_imaginations))
                loss.backward()

                metrics = {"loss/average": loss.item()}
                for i, loss_item in enumerate(loss_imaginations, start=1):
                    metrics[f"loss/imagination_{i}"] = loss_item.item()

                metrics["grad norm"] = (
                    torch.cat(
                        [
                            p.grad.flatten()
                            for p in self.forward_dynamics.model.parameters()
                            if p.grad is not None
                        ]
                    )
                    .norm()
                    .item()
                )
                mlflow.log_metrics(
                    {f"forward-dynamics/{k}": v for k, v in metrics.items()},
                    self.global_step,
                )
                self.optimizers[OPTIMIZER_NAME].step()
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
