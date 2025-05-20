from functools import partial
from pathlib import Path

import pytest
import torch
from pamiq_core.data.impls import RandomReplacementBuffer
from pamiq_core.testing import connect_components
from pamiq_core.torch import TorchTrainingModel
from pytest_mock import MockerFixture
from tensordict import TensorDict
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sample.data import BufferName, DataKey
from sample.models import ModelName
from sample.models.temporal_encoder import ObsInfo, TemporalEncoder
from sample.trainers.sampler import RandomTimeSeriesSampler
from sample.trainers.temporal_encoder import (
    TemporalEncoderTrainer,
    transpose_and_stack_collator,
)
from tests.sample.helpers import parametrize_device


class TestTransposeAndStackCollator:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_tensor_batch(self, batch_size: int):
        """Test collation of tensor batches."""
        # Create sample batch with observation and hidden state
        batch_items = [
            (torch.randn(3, 32, 32), torch.randn(8, 64)) for _ in range(batch_size)
        ]

        result = transpose_and_stack_collator(batch_items)

        assert isinstance(result, tuple)
        assert len(result) == 2
        # Check observation tensor
        assert result[0].shape == (batch_size, 3, 32, 32)
        # Check hidden state tensor
        assert result[1].shape == (batch_size, 8, 64)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_tensordict_batch(self, batch_size: int):
        """Test collation of tensordict batches."""
        # Create sample batch with multimodal observations
        batch_items = [
            (
                TensorDict(
                    {
                        "image": torch.randn(3, 32, 32),
                        "audio": torch.randn(16),
                    },
                    [],
                ),
                torch.randn(8, 64),  # hidden state
            )
            for _ in range(batch_size)
        ]

        result = transpose_and_stack_collator(batch_items)

        assert isinstance(result, tuple)
        assert len(result) == 2
        # Check observation tensordict
        assert isinstance(result[0], TensorDict)
        assert result[0]["image"].shape == (batch_size, 3, 32, 32)
        assert result[0]["audio"].shape == (batch_size, 16)
        # Check hidden state tensor
        assert result[1].shape == (batch_size, 8, 64)


class TestTemporalEncoderTrainer:
    SEQ_LEN = 5
    DEPTH = 2
    DIM = 8
    OBS_INFOS = {
        "image": ObsInfo(dim=32, dim_hidden=16, num_tokens=4),
        "audio": ObsInfo(dim=24, dim_hidden=12, num_tokens=3),
    }

    @pytest.fixture
    def temporal_encoder(self):
        return TemporalEncoder(self.OBS_INFOS, self.DIM, self.DEPTH, self.DIM * 2, 0.1)

    @pytest.fixture
    def models(self, temporal_encoder):
        return {ModelName.TEMPORAL_ENCODER: temporal_encoder}

    @pytest.fixture
    def data_buffers(self):
        return {
            BufferName.TEMPORAL: RandomReplacementBuffer(
                [DataKey.OBSERVATION, DataKey.HIDDEN], max_size=16
            )
        }

    @pytest.fixture
    def partial_dataloader(self):
        return partial(DataLoader, batch_size=2)

    @pytest.fixture
    def partial_sampler(self):
        return partial(RandomTimeSeriesSampler, sequence_length=self.SEQ_LEN)

    @pytest.fixture
    def partial_optimizer(self):
        return partial(AdamW, lr=1e-4)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_sampler,
        partial_optimizer,
        mocker: MockerFixture,
    ):
        mocker.patch("sample.trainers.temporal_encoder.mlflow")
        return TemporalEncoderTrainer(
            partial_dataloader,
            partial_sampler,
            partial_optimizer,
            min_buffer_size=2,
            min_new_data_count=1,
        )

    @parametrize_device
    def test_run(self, device, data_buffers, models, trainer: TemporalEncoderTrainer):
        """Test Temporal Encoder Trainer workflow."""
        models = {
            name: TorchTrainingModel(m, has_inference_model=False, device=device)
            for name, m in models.items()
        }

        components = connect_components(
            trainers=trainer, buffers=data_buffers, models=models
        )
        collector = components.data_collectors[BufferName.TEMPORAL]

        # Collect temporal data
        for _ in range(8):
            observations = TensorDict(
                {
                    k: torch.randn(v.num_tokens, v.dim)
                    for k, v in self.OBS_INFOS.items()
                },
                batch_size=(),
            )
            hidden = torch.randn(self.DEPTH, self.DIM)

            collector.collect(
                {DataKey.OBSERVATION: observations, DataKey.HIDDEN: hidden}
            )

        assert trainer.global_step == 0
        assert trainer.run() is True
        assert trainer.global_step > 0
        global_step = trainer.global_step
        assert trainer.run() is False
        assert trainer.global_step == global_step

    def test_save_and_load_state(self, trainer: TemporalEncoderTrainer, tmp_path: Path):
        trainer_path = tmp_path / "trainer"
        trainer.save_state(trainer_path)
        global_step = trainer.global_step
        assert (trainer_path / "global_step").is_file()

        trainer.global_step = -1
        trainer.load_state(trainer_path)
        assert trainer.global_step == global_step
