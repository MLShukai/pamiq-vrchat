import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import rootutils

PROJECT_ROOT = rootutils.setup_root(__file__)  # Retrieve project root directory


# #########################################
#      Environment Hyper parameters
# #########################################


class InteractionHParams:
    """Environment hyper parameter name space."""

    frame_interval: float = 0.1  # seconds

    class Env:
        class Obs:
            class Image:
                size: tuple[int, int] = (144, 144)  # (height, width)
                channels: int = 3

            class Audio:
                sample_rate: int = 16000
                channel_size: int = 2
                frame_size: int = 16080

        class Action:
            class Mouse:
                time_constant: float = 0.2

            class Osc:
                host: str = "127.0.0.1"
                port: int = 9000
                time_constant: float = 0.2

    class Agent:
        imagination_length: int = 7


# #########################################
#          Model Hyper Parameters
# #########################################


@dataclass
class ModelHParams:
    """Model hyper parameter name space.

    Default values are large size model
    """

    @dataclass
    class ImageJEPA:
        patch_size: tuple[int, int] = (12, 12)
        hidden_dim: int = 432
        embed_dim: int = 128
        depth: int = 6
        num_heads: int = 3
        output_downsample: int = 3

    @dataclass
    class AudioJEPA: ...

    @dataclass
    class TemporalEncoder:
        image_dim: int = 2048
        dim: int = 1024
        depth: int = 8
        dim_ff_hidden: int = dim * 4
        dropout: float = 0.1

    @dataclass
    class ForwardDynamics:
        action_dim: int = 8
        dim: int = 1024
        depth: int = 8
        dim_ff_hidden: int = dim * 4
        dropout: float = 0.1

    @dataclass
    class Policy:
        dim: int = 1024
        depth: int = 8
        dim_ff_hidden: int = dim * 4
        dropout: float = 0.1

    image_jepa: ImageJEPA
    audio_jepa: AudioJEPA
    temporal_encoder: TemporalEncoder
    forward_dynamics: ForwardDynamics
    policy: Policy

    @classmethod
    def create_large(cls) -> Self:
        """Create large size model hyper parameters."""
        return cls(
            image_jepa=cls.ImageJEPA(),
            audio_jepa=cls.AudioJEPA(),
            temporal_encoder=cls.TemporalEncoder(),
            forward_dynamics=cls.ForwardDynamics(),
            policy=cls.Policy(),
        )


# #########################################
#         Trainer Hyper Parameters
# #########################################


@dataclass
class TrainerHParams:
    """Trainer hyper parameter name space."""

    class ImageJEPA: ...

    class AudioJEPA: ...

    class TemporalEncoder:
        lr: float = 0.0001
        seq_len: int = 256
        # Iteration count is max_samples / batch_size
        max_samples: int = 64
        batch_size: int = 1
        min_new_data_count: int = 128

    class ForwardDynamics:
        lr: float = 0.0001
        seq_len: int = 256
        # Iteration count is max_samples / batch_size
        max_samples: int = 64
        batch_size: int = 1
        min_new_data_count: int = 128

    class PPOPolicy:
        lr: float = 0.0001
        seq_len: int = 256
        # Iteration count is max_samples / batch_size
        max_samples: int = 64
        batch_size: int = 1
        min_new_data_count: int = 128


# #############################################
#          Data Buffer Hyper Parameters
# #############################################


class DataBufferHParams:
    """DataBuffer hyper parameter namespace."""

    class Image: ...

    class Audio: ...

    class Temporal:
        max_size = 1000

    class ForwardDynamics:
        max_size = 1000

    class Policy:
        max_size = 1000


# #############################################
#               Launch Arguments
# #############################################


@dataclass
class CliArgs:
    """Arguments for launch."""

    model_size: Literal["tiny", "small", "medium", "large"]
    """Model size selection."""

    device: str = "cuda"
    """Compute device for model."""

    output_dir: Path = PROJECT_ROOT / "logs"
    """Root directory to store states and logs."""


# #############################################
#                Main procedure
# #############################################

import logging
from datetime import datetime

import colorlog
import mlflow
import torch
import tyro
from pamiq_core import (
    DataBuffer,
    Interaction,
    LaunchConfig,
    Trainer,
    TrainingModel,
    launch,
)

from pamiq_vrchat import ActionType, ObservationType
from sample.data import BufferName
from sample.models import ModelName
from sample.utils import average_exponentially


def main() -> None:
    # #########################################
    #              Read Cli Args
    # #########################################

    args = tyro.cli(CliArgs)

    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # #########################################
    #               Setup Logging
    # #########################################

    stream_handler = colorlog.StreamHandler()
    stream_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
    )
    file_handler = logging.handlers.TimedRotatingFileHandler(
        args.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"),
        when="D",
        backupCount=6,  # 7 days logs
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])

    # #########################################
    #         Create Hparams Objects
    # #########################################

    match args.model_size:
        case "large":
            model_hparams = ModelHParams.create_large()
        case _:
            raise

    # #########################################
    #       Create Interaction Components
    # #########################################

    def create_interaction() -> Interaction:
        from pamiq_core import FixedIntervalInteraction
        from pamiq_core.interaction.modular_env import (
            ActuatorsDict,
            ModularEnvironment,
            SensorsDict,
        )
        from pamiq_core.interaction.wrappers import ActuatorWrapper, SensorWrapper

        from pamiq_vrchat import actuators, sensors
        from sample import transforms
        from sample.agents import (
            CuriosityAgent,
            IntegratedCuriosityFramework,
            TemporalEncodingAgent,
            UnimodalEncodingAgent,
        )

        # ----- Agent -----
        agent = IntegratedCuriosityFramework(
            unimodal_agents={
                ObservationType.IMAGE: UnimodalEncodingAgent(
                    ModelName.IMAGE_JEPA_TARGET_ENCODER, BufferName.IMAGE
                ),
                ObservationType.AUDIO: UnimodalEncodingAgent(
                    ModelName.AUDIO_JEPA_TARGET_ENCODER, BufferName.AUDIO
                ),
            },
            temporal_agent=TemporalEncodingAgent(
                torch.zeros(
                    model_hparams.temporal_encoder.depth,
                    model_hparams.temporal_encoder.dim,
                    device=device,
                )
            ),
            curiosity_agent=CuriosityAgent(
                initial_forward_dynamics_hidden=torch.zeros(
                    model_hparams.forward_dynamics.depth,
                    model_hparams.forward_dynamics.dim,
                    device=device,
                ),
                initial_policy_hidden=torch.zeros(
                    model_hparams.policy.depth,
                    model_hparams.policy.dim,
                ),
                max_imagination_steps=InteractionHParams.Agent.imagination_length,
                reward_average_method=average_exponentially,
                log_every_n_steps=round(1 / InteractionHParams.frame_interval),  # 1 sec
            ),
        )

        # ----- Environment -----
        hparams = InteractionHParams.Env
        environment = ModularEnvironment(
            sensor=SensorsDict(
                {
                    ObservationType.IMAGE: SensorWrapper(
                        sensors.ImageSensor(),
                        transforms.image.create_transform(
                            hparams.Obs.Image.size,
                        ),
                    ),
                    ObservationType.AUDIO: SensorWrapper(
                        sensors.AudioSensor(
                            frame_size=int(44100 * InteractionHParams.frame_interval),
                            sample_rate=44100,
                            channels=2,
                        ),
                        transforms.audio.create_transform(
                            source_sample_rate=44100,
                            target_sample_rate=hparams.Obs.Audio.sample_rate,
                            target_frame_size=hparams.Obs.Audio.frame_size,
                        ),
                    ),
                }
            ),
            actuator=ActuatorWrapper(
                ActuatorsDict(
                    {
                        ActionType.MOUSE: actuators.SmoothMouseActuator(
                            InteractionHParams.frame_interval,
                            hparams.Action.Mouse.time_constant,
                        ),
                        ActionType.OSC: actuators.SmoothOscActuator(
                            hparams.Action.Osc.host,
                            hparams.Action.Osc.port,
                            delta_time=InteractionHParams.frame_interval,
                            time_constant=hparams.Action.Osc.time_constant,
                        ),
                    }
                ),
                transforms.action.ActionTransform(),
            ),
        )
        return FixedIntervalInteraction.with_sleep_adjustor(
            agent, environment, InteractionHParams.frame_interval
        )

    # #########################################
    #          Create Model Components
    # #########################################

    def create_models() -> dict[str, TrainingModel[Any]]:
        from pamiq_core.torch import TorchTrainingModel

        from sample.models import (
            ForwardDynamics,
            PolicyValueCommon,
            TemporalEncoder,
            create_image_jepa,
        )
        from sample.models.temporal_encoder import ObsInfo as TemporalEncoderObsInfo
        from sample.transforms.action import ACTION_CHOICES

        temporal_encoder_obs_infos: dict[str, TemporalEncoderObsInfo] = {}

        # ----- Image JEPA -----
        hparams = model_hparams.image_jepa
        context_encoder, target_encoder, predictor, infer = create_image_jepa(
            image_size=InteractionHParams.Env.Obs.Image.size,
            patch_size=hparams.patch_size,
            in_channels=InteractionHParams.Env.Obs.Image.channels,
            hidden_dim=hparams.hidden_dim,
            embed_dim=hparams.embed_dim,
            depth=hparams.depth,
            num_heads=hparams.num_heads,
            output_downsample=hparams.output_downsample,
        )
        image_jepa_context_encoder = TorchTrainingModel(
            context_encoder,
            has_inference_model=False,
            device=device,
        )

        image_jepa_target_encoder = TorchTrainingModel(
            target_encoder,
            has_inference_model=True,
            inference_procedure=infer,
            device=device,
        )
        image_jepa_predictor = TorchTrainingModel(
            predictor,
            has_inference_model=False,
            device=device,
        )

        temporal_encoder_obs_infos[ObservationType.IMAGE] = TemporalEncoderObsInfo(
            dim=hparams.embed_dim,
            dim_hidden=ModelHParams.TemporalEncoder.image_dim,
            num_tokens=infer.output_patch_count,
        )

        # ----- Temporal Encoder -----
        hparams = model_hparams.temporal_encoder
        temporal_encoder = TorchTrainingModel(
            TemporalEncoder(
                obs_infos=temporal_encoder_obs_infos,
                dim=hparams.dim,
                depth=hparams.depth,
                dim_ff_hidden=hparams.dim_ff_hidden,
                dropout=hparams.dropout,
            ),
            has_inference_model=True,
            device=device,
            inference_procedure=TemporalEncoder.infer,
        )

        # ----- Forward Dynamics -----
        hparams = model_hparams.forward_dynamics
        forward_dynamics = TorchTrainingModel(
            ForwardDynamics(
                obs_dim=model_hparams.temporal_encoder.dim,
                action_choices=list(ACTION_CHOICES),
                action_dim=hparams.action_dim,
                dim=hparams.dim,
                depth=hparams.depth,
                dim_ff_hidden=hparams.dim_ff_hidden,
                dropout=hparams.dropout,
            ),
            has_inference_model=True,
            device=device,
        )

        # ----- Policy -----
        hparams = model_hparams.policy
        policy = TorchTrainingModel(
            PolicyValueCommon(
                obs_dim=model_hparams.temporal_encoder.dim,
                action_choices=list(ACTION_CHOICES),
                dim=hparams.dim,
                depth=hparams.depth,
                dim_ff_hidden=hparams.dim_ff_hidden,
                dropout=hparams.dropout,
            )
        )

        return {
            ModelName.IMAGE_JEPA_CONTEXT_ENCODER: image_jepa_context_encoder,
            ModelName.IMAGE_JEPA_TARGET_ENCODER: image_jepa_target_encoder,
            ModelName.IMAGE_JEPA_PREDICTOR: image_jepa_predictor,
            ModelName.TEMPORAL_ENCODER: temporal_encoder,
            ModelName.FORWARD_DYNAMICS: forward_dynamics,
            ModelName.POLICY_VALUE: policy,
        }

    # #########################################
    #             Create Trainers
    # #########################################

    def create_trainers() -> dict[str, Trainer]:
        from functools import partial

        from torch.optim import AdamW

        from sample.trainers import (
            ImaginingForwardDynamicsTrainer,
            PPOPolicyTrainer,
            TemporalEncoderTrainer,
        )

        # ----- Temporal Encoder Trainer -----
        hparams = TrainerHParams.TemporalEncoder
        temporal_encoder = TemporalEncoderTrainer(
            partial_optimzier=partial(AdamW, lr=hparams.lr),
            seq_len=hparams.seq_len,
            max_samples=hparams.seq_len,
            batch_size=hparams.batch_size,
            min_new_data_count=hparams.min_new_data_count,
            min_buffer_size=hparams.seq_len + 1,
        )

        # ----- Forward Dynamics Trainer -----
        hparams = TrainerHParams.ForwardDynamics
        forward_dynamics = ImaginingForwardDynamicsTrainer(
            partial_optimizer=partial(AdamW, lr=hparams.lr),
            seq_len=hparams.seq_len,
            max_samples=hparams.max_samples,
            batch_size=hparams.batch_size,
            imagination_length=InteractionHParams.Agent.imagination_length,
            min_buffer_size=(
                hparams.seq_len + InteractionHParams.Agent.imagination_length
            ),
            min_new_data_count=hparams.min_new_data_count,
            imagination_average_method=average_exponentially,
        )

        # ----- PPO Policy Trainer -----
        hparams = TrainerHParams.PPOPolicy
        policy = PPOPolicyTrainer(
            partial_optimizer=partial(AdamW, lr=hparams.lr),
            seq_len=hparams.seq_len,
            max_samples=hparams.max_samples,
            batch_size=hparams.batch_size,
            min_buffer_size=hparams.seq_len,
            min_new_data_count=hparams.min_new_data_count,
        )

        return {
            "temporal_encoder": temporal_encoder,
            "forward_dynamics": forward_dynamics,
            "policy": policy,
        }

    # #########################################
    #            Create Data Buffers
    # #########################################

    def create_data_buffers() -> dict[str, DataBuffer[Any]]:
        from pamiq_core.data.impls import SequentialBuffer

        from sample.data import DataKey

        # TODO: Write Data Buffer definition for Audio and Image.

        # ----- Temporal Buffer -----
        temporal = SequentialBuffer(
            collecting_data_names=[DataKey.OBSERVATION, DataKey.HIDDEN],
            max_size=DataBufferHParams.Temporal.max_size,
        )

        # ----- Forward Dynamics Buffer -----
        forward_dynamics = SequentialBuffer(
            collecting_data_names=[DataKey.OBSERVATION, DataKey.ACTION, DataKey.HIDDEN],
            max_size=DataBufferHParams.ForwardDynamics.max_size,
        )

        # ----- Policy Buffer -----
        policy = SequentialBuffer(
            collecting_data_names=[
                DataKey.OBSERVATION,
                DataKey.HIDDEN,
                DataKey.ACTION,
                DataKey.ACTION_LOG_PROB,
                DataKey.REWARD,
                DataKey.VALUE,
            ],
            max_size=DataBufferHParams.Policy.max_size,
        )

        return {
            BufferName.TEMPORAL: temporal,
            BufferName.FORWARD_DYNAMICS: forward_dynamics,
            BufferName.POLICY: policy,
        }

    # #########################################
    #                 Launch
    # #########################################

    mlflow.set_tracking_uri(args.output_dir / "mlflow")

    with mlflow.start_run():
        launch(
            interaction=create_interaction(),
            models=create_models(),
            data=create_data_buffers(),
            trainers=create_trainers(),
            config=LaunchConfig(
                states_dir=args.output_dir / "states",
                save_state_interval=24 * 60 * 60,  # 1 day.
                max_keep_states=3,
            ),
        )


if __name__ == "__main__":
    main()
