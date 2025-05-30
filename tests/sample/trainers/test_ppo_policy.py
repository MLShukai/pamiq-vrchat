from functools import partial
from pathlib import Path

import pytest
import torch
from pamiq_core.data.impls import SequentialBuffer
from pamiq_core.testing import connect_components
from pamiq_core.torch import TorchTrainingModel
from pytest_mock import MockerFixture
from torch.optim import AdamW

from sample.data import BufferName, DataKey
from sample.models import ModelName
from sample.models.policy import PolicyValueCommon
from sample.trainers.ppo_policy import PPOPolicyTrainer
from tests.sample.helpers import parametrize_device


class TestPPOPolicyTrainer:
    DEPTH = 2
    DIM = 8
    OBS_DIM = 16
    ACTION_CHOICES = [3, 4]  # Multiple discrete actions
    SEQ_LEN = 10

    @pytest.fixture
    def policy_value_model(self):
        return PolicyValueCommon(
            self.OBS_DIM, self.ACTION_CHOICES, self.DIM, self.DEPTH, self.DIM * 2
        )

    @pytest.fixture
    def models(self, policy_value_model):
        return {ModelName.POLICY_VALUE: policy_value_model}

    @pytest.fixture
    def data_buffers(self):
        return {
            BufferName.POLICY: SequentialBuffer(
                [
                    DataKey.OBSERVATION,
                    DataKey.HIDDEN,
                    DataKey.ACTION,
                    DataKey.ACTION_LOG_PROB,
                    DataKey.REWARD,
                    DataKey.VALUE,
                ],
                max_size=32,
            )
        }

    @pytest.fixture
    def trainer(
        self,
        mocker: MockerFixture,
    ):
        mocker.patch("sample.trainers.ppo_policy.mlflow")
        return PPOPolicyTrainer(
            partial_optimizer=partial(AdamW, lr=3e-4),
            seq_len=self.SEQ_LEN,
            max_samples=4,
            batch_size=2,
            min_buffer_size=3,
            min_new_data_count=1,
        )

    @parametrize_device
    def test_run(self, device, data_buffers, models, trainer: PPOPolicyTrainer):
        """Test PPO Policy Trainer workflow."""
        models = {
            name: TorchTrainingModel(m, has_inference_model=False, device=device)
            for name, m in models.items()
        }

        components = connect_components(
            trainers=trainer, buffers=data_buffers, models=models
        )
        collector = components.data_collectors[BufferName.POLICY]

        # Collect policy data
        for _ in range(20):
            observations = torch.randn(self.OBS_DIM)
            hidden = torch.randn(self.DEPTH, self.DIM)
            actions = torch.stack(
                [torch.randint(0, dim, ()) for dim in self.ACTION_CHOICES], dim=-1
            )
            action_log_probs = torch.randn(len(self.ACTION_CHOICES))
            rewards = torch.randn(())
            values = torch.randn(())

            collector.collect(
                {
                    DataKey.OBSERVATION: observations,
                    DataKey.HIDDEN: hidden,
                    DataKey.ACTION: actions,
                    DataKey.ACTION_LOG_PROB: action_log_probs,
                    DataKey.REWARD: rewards,
                    DataKey.VALUE: values,
                }
            )

        assert trainer.global_step == 0
        assert trainer.run() is True
        assert trainer.global_step > 0
        global_step = trainer.global_step
        assert trainer.run() is False
        assert trainer.global_step == global_step

    def test_save_and_load_state(self, trainer: PPOPolicyTrainer, tmp_path: Path):
        trainer_path = tmp_path / "trainer"
        trainer.save_state(trainer_path)
        global_step = trainer.global_step
        assert (trainer_path / "global_step").is_file()

        trainer.global_step = -1
        trainer.load_state(trainer_path)
        assert trainer.global_step == global_step
