from functools import partial
from pathlib import Path

import pytest
import torch
from pamiq_core.data.impls import SequentialBuffer
from pamiq_core.testing import connect_components
from pamiq_core.torch import TorchTrainingModel
from pytest_mock import MockerFixture
from torch.nn import Linear
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sample.data import BufferName, DataKey
from sample.models import ModelName
from sample.models.components.fc_scalar_head import FCScalarHead
from sample.models.components.multi_discretes import FCMultiCategoricalHead
from sample.models.components.qlstm import QLSTM
from sample.models.policy import PolicyValueCommon
from sample.trainers.ppo_policy import PPOPolicyTrainer
from sample.trainers.sampler import RandomTimeSeriesSampler
from tests.sample.helpers import parametrize_device


class TestPPOPolicyTrainer:
    HIDDEN_DEPTH = 2
    HIDDEN_DIM = 8
    OBS_DIM = 16
    ACTION_DIMS = [3, 4]  # Multiple discrete actions
    SEQ_LEN = 10

    @pytest.fixture
    def policy_value_model(self):
        observation_flatten = Linear(self.OBS_DIM, self.HIDDEN_DIM)
        core_model = QLSTM(
            depth=self.HIDDEN_DEPTH,
            dim=self.HIDDEN_DIM,
            dim_ff_hidden=self.HIDDEN_DIM * 2,
            dropout=0.0,
        )
        policy_head = FCMultiCategoricalHead(self.HIDDEN_DIM, self.ACTION_DIMS)
        value_head = FCScalarHead(self.HIDDEN_DIM, squeeze_scalar_dim=True)

        return PolicyValueCommon(
            observation_flatten=observation_flatten,
            core_model=core_model,
            policy_head=policy_head,
            value_head=value_head,
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
    def partial_dataloader(self):
        return partial(DataLoader, batch_size=4)

    @pytest.fixture
    def partial_sampler(self):
        return partial(RandomTimeSeriesSampler, sequence_length=self.SEQ_LEN)

    @pytest.fixture
    def partial_optimizer(self):
        return partial(AdamW, lr=3e-4)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_sampler,
        partial_optimizer,
        mocker: MockerFixture,
    ):
        mocker.patch("sample.trainers.ppo_policy.mlflow")
        return PPOPolicyTrainer(
            partial_dataloader,
            partial_sampler,
            partial_optimizer,
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
            hidden = torch.randn(self.HIDDEN_DEPTH, self.HIDDEN_DIM)
            actions = torch.stack(
                [torch.randint(0, dim, ()) for dim in self.ACTION_DIMS], dim=-1
            )
            action_log_probs = torch.randn(len(self.ACTION_DIMS))
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
