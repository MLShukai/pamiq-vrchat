from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pamiq_core.data.impls import SequentialBuffer
from pamiq_core.testing import connect_components
from pamiq_core.torch import TorchTrainingModel
from pytest_mock import MockerFixture
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sample.data import BufferName, DataKey
from sample.models import ModelName
from sample.models.components.deterministic_normal import FCDeterministicNormalHead
from sample.models.components.qlstm import QLSTM
from sample.models.forward_dynamics import ForwardDynamics
from sample.trainers.forward_dynamics import ImaginingForwardDynamicsTrainer
from sample.trainers.sampler import RandomTimeSeriesSampler
from tests.sample.helpers import parametrize_device


class TestImaginingForwardDynamicsTrainer:
    BATCH = 4
    DEPTH = 8
    DIM = 16
    DIM_FF_HIDDEN = 32
    LEN = 64
    LEN_SEQ = 16
    DROPOUT = 0.1
    DIM_OBS = 32
    DIM_ACTION = 8
    ACTION_CHOICES = [4, 9, 2]

    @pytest.fixture
    def forward_dynamics(
        self,
    ):
        return ForwardDynamics(
            self.DIM_OBS,
            self.ACTION_CHOICES,
            self.DIM_ACTION,
            self.DIM,
            self.DEPTH,
            self.DIM * 2,
        )

    @pytest.fixture
    def data_buffers(self):
        return {
            BufferName.FORWARD_DYNAMICS: SequentialBuffer(
                [DataKey.OBSERVATION, DataKey.ACTION, DataKey.HIDDEN], max_size=self.LEN
            )
        }

    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=2)
        return partial_dataloader

    @pytest.fixture
    def partial_sampler(self):
        partial_sampler = partial(RandomTimeSeriesSampler, sequence_length=self.LEN_SEQ)
        return partial_sampler

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(AdamW, lr=1e-4, weight_decay=0.04)
        return partial_optimizer

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_sampler,
        partial_optimizer,
        mocker: MockerFixture,
    ):
        mocker.patch("sample.trainers.forward_dynamics.mlflow")
        trainer = ImaginingForwardDynamicsTrainer(
            partial_dataloader,
            partial_sampler,
            partial_optimizer,
            imagination_length=4,
            min_buffer_size=8,
            min_new_data_count=4,
        )
        return trainer

    def create_action(self) -> torch.Tensor:
        actions = []
        for choice in self.ACTION_CHOICES:
            actions.append(torch.randint(0, choice, ()))
        return torch.stack(actions, dim=-1)

    @parametrize_device
    def test_run(self, device, data_buffers, forward_dynamics, trainer):
        models = {
            str(ModelName.FORWARD_DYNAMICS): TorchTrainingModel(
                forward_dynamics, has_inference_model=False, device=device
            ),
        }
        components = connect_components(
            trainers=trainer, buffers=data_buffers, models=models
        )
        collector = components.data_collectors[BufferName.FORWARD_DYNAMICS]
        for _ in range(self.LEN):
            collector.collect(
                {
                    DataKey.OBSERVATION: torch.randn(self.DIM_OBS),
                    DataKey.ACTION: self.create_action(),
                    DataKey.HIDDEN: torch.randn(self.DEPTH, self.DIM),
                }
            )

        assert trainer.global_step == 0
        assert trainer.run() is True
        assert trainer.global_step > 0
        global_step = trainer.global_step
        assert trainer.run() is False
        assert trainer.global_step == global_step

    def test_save_and_load_state(self, trainer, tmp_path: Path):
        trainer_path = tmp_path / "trainer"
        trainer.save_state(trainer_path)
        global_step = trainer.global_step
        assert (trainer_path / "global_step").is_file()

        trainer.global_step = -1
        trainer.load_state(trainer_path)
        assert trainer.global_step == global_step
