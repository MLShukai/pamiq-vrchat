from dataclasses import dataclass
from typing import Any, Literal, Self

# #########################################
#      Environment Hyper parameters
# #########################################


class InteractionHPs:
    """Environment hyper parameter name space."""

    frame_interval: float = 0.1  # seconds

    class Env:
        class Obs:
            class Image:
                height: int = 144
                width: int = 144
                channels: int = 3

                @classmethod
                def size(cls) -> tuple[int, int]:
                    return (cls.height, cls.width)

            class Audio:
                sample_rate: int = 16000
                channel_size: int = 2
                sample_size: int = 16080

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
class ModelHPs:
    """Model hyper parameter name space.

    Default values are large size model
    """

    @dataclass
    class ImageJEPA: ...

    @dataclass
    class AudioJEPA: ...

    @dataclass
    class TemporalEncoder:
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
class TrainerHPs:
    """Trainer hyper parameter name space."""

    @dataclass
    class ImageJEPA: ...

    @dataclass
    class AudioJEPA: ...

    @dataclass
    class TemporalEncoder: ...

    @dataclass
    class ForwardDynamics: ...

    @dataclass
    class PPOPolicy: ...


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


# #############################################
#                Main procedure
# #############################################

import torch
import tyro
from pamiq_core import Interaction, TrainingModel

from sample.data import BufferName
from sample.models import ModelName
from sample.utils import average_exponentially


def main() -> None:
    args = tyro.cli(CliArgs)

    device = torch.device(args.device)
    # #########################################
    #         Create Hparams Objects
    # #########################################

    match args.model_size:
        case "large":
            model_hparams = ModelHPs.create_large()
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

        from pamiq_vrchat import ActionType, ObservationType, actuators, sensors
        from sample import transforms
        from sample.agents import (
            CuriosityAgent,
            IntegratedCuriosityFramework,
            TemporalEncodingAgent,
            UnimodalEncodingAgent,
        )

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
                max_imagination_steps=InteractionHPs.Agent.imagination_length,
                reward_average_method=average_exponentially,
                log_every_n_steps=round(1 / InteractionHPs.frame_interval),  # 1 sec
            ),
        )

        environment = ModularEnvironment(
            sensor=SensorsDict(
                {
                    ObservationType.IMAGE: SensorWrapper(
                        sensors.ImageSensor(),
                        transforms.image.create_vrchat_transform(
                            InteractionHPs.Env.Obs.Image.size(),
                        ),
                    )
                }
            ),
            actuator=ActuatorWrapper(
                ActuatorsDict(
                    {
                        ActionType.MOUSE: actuators.SmoothMouseActuator(
                            InteractionHPs.frame_interval,
                            InteractionHPs.Env.Action.Mouse.time_constant,
                        ),
                        ActionType.OSC: actuators.SmoothOscActuator(
                            InteractionHPs.Env.Action.Osc.host,
                            InteractionHPs.Env.Action.Osc.port,
                            delta_time=InteractionHPs.frame_interval,
                            time_constant=InteractionHPs.Env.Action.Osc.time_constant,
                        ),
                    }
                ),
                transforms.action.ActionTransform(),
            ),
        )
        return FixedIntervalInteraction.with_sleep_adjustor(
            agent, environment, InteractionHPs.frame_interval
        )

    # #########################################
    #          Create Model Components
    # #########################################

    def create_models(self) -> dict[ModelName, TrainingModel[Any]]:
        from pamiq_core.torch import TorchTrainingModel

        from sample.models import ForwardDynamics, PolicyValueCommon, TemporalEncoder
        from sample.transforms.action import ACTION_CHOICES

        # TODO: create JEPA and assign obs infos
        temporal_encoder = TorchTrainingModel(
            TemporalEncoder(
                obs_infos={},
                dim=model_hparams.temporal_encoder.dim,
                depth=model_hparams.temporal_encoder.depth,
                dim_ff_hidden=model_hparams.temporal_encoder.dim_ff_hidden,
                dropout=model_hparams.temporal_encoder.dropout,
            ),
            has_inference_model=True,
            device=device,
            inference_procedure=TemporalEncoder.infer,
        )

        forward_dynamics = TorchTrainingModel(
            ForwardDynamics(
                obs_dim=model_hparams.temporal_encoder.dim,
                action_choices=list(ACTION_CHOICES),
                action_dim=model_hparams.forward_dynamics.action_dim,
                dim=model_hparams.forward_dynamics.dim,
                depth=model_hparams.forward_dynamics.depth,
                dim_ff_hidden=model_hparams.forward_dynamics.dim_ff_hidden,
                dropout=model_hparams.forward_dynamics.dropout,
            ),
            has_inference_model=True,
            device=device,
        )

        policy = TorchTrainingModel(
            PolicyValueCommon(
                obs_dim=model_hparams.temporal_encoder.dim,
                action_choices=list(ACTION_CHOICES),
                dim=model_hparams.policy.dim,
                depth=model_hparams.policy.depth,
                dim_ff_hidden=model_hparams.policy.dim_ff_hidden,
                dropout=model_hparams.policy.dropout,
            )
        )

        return {
            ModelName.TEMPORAL_ENCODER: temporal_encoder,
            ModelName.FORWARD_DYNAMICS: forward_dynamics,
            ModelName.POLICY_VALUE: policy,
        }
