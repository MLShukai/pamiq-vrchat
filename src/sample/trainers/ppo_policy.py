from functools import partial
from pathlib import Path
from typing import override

import mlflow
import torch
from pamiq_core import DataUser
from pamiq_core.torch import OptimizersSetup, TorchTrainer, get_device
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from sample.data import BufferName, DataKey
from sample.models import ModelName
from sample.models.policy import PolicyValueCommon
from sample.trainers.sampler import RandomTimeSeriesSampler

OPTIMIZER_NAME = "optimizer"


class PPOPolicyTrainer(TorchTrainer):
    """Trainer for policy using Proximal Policy Optimization (PPO)."""

    @override
    def __init__(
        self,
        seq_len: int = 1,
        batch_size: int = 1,
        lr: float = 1e-4,
        max_epochs: int = 1,
        norm_advantage: bool = True,
        clip_coef: float = 0.1,
        entropy_coef: float = 0.0,
        vfunc_coef: float = 0.5,
        data_user_name: str = BufferName.POLICY,
        min_buffer_size: int = 1,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize the PPO Policy trainer.

        Args:
            seq_len: Sequence length per batch.
            batch_size: Number of data samples for 1 step.
            lr: Learning rate for optimizer.
            max_epochs: Maximum number of epochs to train per training session.
            norm_advantage: Whether to normalize advantages.
            clip_coef: Clipping coefficient for PPO.
            entropy_coef: Coefficient for entropy regularization.
            vfunc_coef: Coefficient for value function loss.
            data_user_name: Name of the data user providing training data.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
        """
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.data_user_name = data_user_name
        self.partial_sampler = partial(
            RandomTimeSeriesSampler, sequence_length=seq_len, max_samples=batch_size
        )
        self.partial_optimizer = partial(AdamW, lr=lr)
        self.max_epochs = max_epochs
        self.norm_advantage = norm_advantage
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef
        self.vfunc_coef = vfunc_coef
        self.global_step = 0

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the
        trainer."""
        super().on_data_users_attached()
        self.policy_data_user: DataUser[Tensor] = self.get_data_user(
            self.data_user_name
        )

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer."""
        super().on_training_models_attached()
        self.policy_value = self.get_torch_training_model(
            ModelName.POLICY_VALUE, PolicyValueCommon
        )

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for PPO training.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {
            OPTIMIZER_NAME: self.partial_optimizer(self.policy_value.model.parameters())
        }

    def training_step(self, batch: list[Tensor]) -> dict[str, Tensor]:
        """Perform a single training step on a batch of data."""
        observations, hiddens, actions, action_log_probs, rewards, values = batch

        # Calculate returns (rewards-to-go)
        returns = rewards  # Assuming rewards are already processed as returns
        advantages = returns - values

        # Get new distributions and values
        new_dist, new_values, _ = self.policy_value.model(observations, hiddens[:, 0])
        new_log_probs = new_dist.log_prob(actions)
        entropy = new_dist.entropy()

        # Calculate ratio for PPO
        log_ratio = new_log_probs - action_log_probs
        ratio = log_ratio.exp()

        # Calculate KL divergence and clip fraction
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

        # Normalize advantages
        if self.norm_advantage:
            advantages = advantages / (advantages.std() + 1e-8)

        # Adjust dimensions if needed
        if advantages.ndim < ratio.ndim:
            for _ in range(ratio.ndim - advantages.ndim):
                advantages = advantages.unsqueeze(-1)

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        new_values = new_values.flatten()
        returns = returns.flatten()
        values = values.flatten()

        v_loss_unclipped = (new_values - returns) ** 2
        v_clipped = values + torch.clamp(
            new_values - values, -self.clip_coef, self.clip_coef
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        entropy_loss = entropy.mean()

        # Total loss
        loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.vfunc_coef

        return {
            "loss": loss,
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy_loss,
            "approx_kl": approx_kl,
            "clipfrac": clipfracs,
        }

    @override
    def train(self) -> None:
        """Execute PPO training process."""
        # Get dataset from data user
        data = self.policy_data_user.get_data()

        dataset = TensorDataset(
            *[
                torch.stack(list(data[key]))
                for key in [
                    DataKey.OBSERVATION,
                    DataKey.HIDDEN,
                    DataKey.ACTION,
                    DataKey.ACTION_LOG_PROB,
                    DataKey.REWARD,
                    DataKey.VALUE,
                ]
            ]
        )
        sampler = self.partial_sampler(dataset)
        dataloader = DataLoader(dataset=dataset, sampler=sampler)
        device = get_device(self.policy_value.model)

        for _ in range(self.max_epochs):
            batch: tuple[Tensor, ...]
            for batch in dataloader:
                self.optimizers[OPTIMIZER_NAME].zero_grad()

                # Perform training step
                outputs = self.training_step([d.to(device) for d in batch])
                loss = outputs["loss"]

                # Backward pass
                loss.backward()

                # Calculate gradient norm
                grad_norm = torch.cat(
                    [
                        p.grad.flatten()
                        for p in self.policy_value.model.parameters()
                        if p.grad is not None
                    ]
                ).norm()

                self.optimizers[OPTIMIZER_NAME].step()

                # Logging
                metrics = {k: v.item() for k, v in outputs.items()}
                metrics["grad_norm"] = grad_norm.item()

                mlflow.log_metrics(
                    {
                        f"ppo-policy/{tag}": v.item()
                        if isinstance(v, torch.Tensor)
                        else v
                        for tag, v in metrics.items()
                    },
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
