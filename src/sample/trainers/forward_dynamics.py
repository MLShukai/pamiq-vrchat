from functools import partial
from pathlib import Path
from typing import override

import mlflow
import torch
from pamiq_core import DataUser
from pamiq_core.torch import OptimizersSetup, TorchTrainer, get_device
from pamiq_core.torch.trainer import TorchTrainer
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from sample.data import BufferName, DataKey
from sample.models import ModelName
from sample.models.forward_dynamics import ForwardDynamics
from sample.trainers.sampler import RandomTimeSeriesSampler

OPTIMIZER_NAME = "optimizer"


class ForwardDynamicsTrainer(TorchTrainer):
    """Trainer for the ForwardDynamics model.

    This trainer implements the training loop for the ForwardDynamics
    model, which predicts the next observation distribution given the
    current observation and action. It uses a recurrent core model to
    maintain hidden state across sequential predictions.
    """

    def __init__(
        self,
        partial_dataloader: partial[DataLoader[Tensor]],
        partial_sampler: partial[RandomTimeSeriesSampler],
        partial_optimizer: partial[Optimizer],
        max_epochs: int = 1,
        data_user_name: str = BufferName.FORWARD_DYNAMICS,
        min_buffer_size: int = 0,
        min_new_data_count: int = 0,
    ) -> None:
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.partial_dataloader = partial_dataloader
        self.partial_sampler = partial_sampler
        self.partial_optimizer = partial_optimizer
        self.max_epochs = max_epochs
        self.data_user_name = data_user_name
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
                hiddens = hiddens.to(device)

                observations, hidden, actions, observation_next = (
                    observations[:, :-1],
                    hiddens[:, 0],
                    actions[:, :-1],
                    observations[:, 1:],
                )

                self.optimizers[OPTIMIZER_NAME].zero_grad()
                observation_next_hat_dist, _ = self.forward_dynamics.model(
                    observations, actions, hidden
                )
                loss = -observation_next_hat_dist.log_prob(observation_next).mean()
                loss.backward()
                self.optimizers[OPTIMIZER_NAME].step()
                mlflow.log_metric(
                    "forward-dynamics/loss", loss.item(), step=self.global_step
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
