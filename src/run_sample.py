"""
PAMIQ System on VRChat
======================

This script implements a complete pipeline for training autonomous agents that can interact
with VRChat environments through vision, audio, and action. The system uses a multi-modal
curiosity-driven learning approach.

Architecture Overview:
---------------------
The system consists of four main learning components working in a hierarchical manner:

1. **Unimodal Encoders (Image/Audio JEPA)**: Learn rich representations from raw sensory data
   using Joint Embedding Predictive Architecture, enabling the agent to understand visual
   and auditory patterns.

2. **Temporal Encoder**: Integrates multimodal features across time, capturing temporal
   dependencies and cross-modal relationships essential for understanding dynamic environments.

3. **Forward Dynamics Model**: Predicts future observations given current state and actions.

4. **Policy Network**: Selects actions based on current observations and intrinsic motivation
   derived from prediction errors, driving exploration.

Data Flow:
---------
Raw Sensors → JEPA Encoders → Temporal Integration → Forward Dynamics + Policy → Actions

The system operates in real-time at 10Hz, balancing computational efficiency with responsive
interaction capabilities.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import rootutils

# Retrieve project root directory
PROJECT_ROOT = rootutils.setup_root(__file__)


# ======================================================================================
#                               INTERACTION HYPERPARAMETERS
# ======================================================================================
# These parameters control the real-time interaction between agent and environment.
# They are carefully tuned to balance responsiveness with computational constraints.


class InteractionHParams:
    """Interaction hyperparameter namespace controlling real-time agent-
    environment dynamics.

    These parameters directly impact the agent's ability to perceive and
    respond to the VRChat environment in real-time. The frame interval
    determines the temporal resolution of the interaction loop,
    affecting both learning dynamics and computational load.
    """

    # fmt: off
    frame_interval: float = 0.1  # seconds (10 FPS)

    class Env:
        class Obs:
            class Image:
                size: tuple[int, int] = (144, 144)  # (height, width)
                channels: int = 3  # RGB

            class Audio:
                sample_rate: int = 16000    # Hz
                channel_size: int = 2       # Stereo
                frame_size: int = 16080     # ~1 second at 16kHz, adjusted for patchification

        class Action:
            class Mouse:
                time_constant: float = 0.2  # Smoothing for natural mouse movement
                max_velocity: int = 1000    # pixels/second - Prevents excessive movement

            class Osc:
                host: str = "127.0.0.1"     # VRChat OSC endpoint
                port: int = 9000            # Standard VRChat OSC port
                time_constant: float = 0.2  # Smoothing for natural locomotion movement

    class Agent:
        imagination_length: int = 7 # Steps ahead for imagination.

    # fmt: on


# ======================================================================================
#                              MODEL HYPERPARAMETERS
# ======================================================================================
# These parameters define the neural architecture configurations for each component.
# Different model sizes (tiny/small/medium/large) trade off between performance and
# computational requirements, allowing deployment across various hardware configurations.


@dataclass
class ModelHParams:
    """Model hyperparameter namespace defining neural architecture
    configurations."""

    # fmt: off
    @dataclass
    class ImageJEPA:
        patch_size: tuple[int, int] = (12, 12)  # Patch size for image tokenization
        hidden_dim: int = 768                   # Internal representation dimensionality
        embed_dim: int = 128                    # Output embedding size (compressed representation)
        depth: int = 6                          # Number of transformer layers
        num_heads: int = 4                      # Multi-head attention heads
        output_downsample: int = 3              # Spatial compression factor for efficiency

    @dataclass
    class AudioJEPA:
        hidden_dim: int = 480                   # Smaller than image due to 1D nature of audio
        embed_dim: int = 64                     # Internal representation dimensionality
        depth: int = 6                          # Output embedding size (compressed representation)
        num_heads: int = 3                      # Multi-head attention heads
        output_downsample: int = 2              # Spatial compression factor for efficiency

    @dataclass
    class TemporalEncoder:
        image_dim: int = 1024                   # Projected image feature dimension (internal)
        audio_dim: int = 512                    # Projected audio feature dimension (internal)
        dim: int = 1024                         # Unified multimodal representation size (internal)
        depth: int = 6                          # Layer depth
        dim_ff_expansion_factor: int = 2        # Feed-forward expansion factor (scales `dim`)
        dropout: float = 0.1                    # Regularization to prevent overfitting

    @dataclass
    class ForwardDynamics:
        action_dim: int = 8                     # Action embedding dimensionality
        dim: int = 1536                         # Large capacity for complex dynamics
        depth: int = 8                          # Layer depth
        dim_ff_expansion_factor: int = 4        # Feed-forward expansion factor (scales `dim`)
        dropout: float = 0.1                    # Regularization to prevent overfitting

    @dataclass
    class Policy:
        dim: int = 1536                         # Large capacity for complex dynamics
        depth: int = 8                          # Layer depth
        dim_ff_expansion_factor: int = 4        # Feed-forward expansion factor (scales `dim`)
        dropout: float = 0.1                    # Regularization to prevent overfitting
    # fmt: on

    image_jepa: ImageJEPA
    audio_jepa: AudioJEPA
    temporal_encoder: TemporalEncoder
    forward_dynamics: ForwardDynamics
    policy: Policy

    @classmethod
    def create_huge(cls) -> Self:
        """Create large model configuration.

        VRAM usage is ~23GiB (bfloat16)
        """
        return cls(
            image_jepa=cls.ImageJEPA(hidden_dim=1024, depth=8, num_heads=8),
            audio_jepa=cls.AudioJEPA(hidden_dim=512, depth=8, num_heads=4),
            temporal_encoder=cls.TemporalEncoder(dim=1536),
            forward_dynamics=cls.ForwardDynamics(dim=2048, depth=10),
            policy=cls.Policy(dim=2048, depth=10),
        )

    @classmethod
    def create_large(cls) -> Self:
        """Create large model configuration.

        VRAM usage is ~12GiB (bfloat16)
        """
        return cls(
            image_jepa=cls.ImageJEPA(),
            audio_jepa=cls.AudioJEPA(),
            temporal_encoder=cls.TemporalEncoder(),
            forward_dynamics=cls.ForwardDynamics(),
            policy=cls.Policy(),
        )

    @classmethod
    def create_medium(cls) -> Self:
        """Create medium model configuration.

        VRAM usage is ~6.5GiB. (bfloat16)
        """
        return cls(
            image_jepa=cls.ImageJEPA(
                hidden_dim=432,
                depth=6,
                num_heads=3,
            ),
            audio_jepa=cls.AudioJEPA(
                hidden_dim=320,
                depth=6,
                num_heads=2,
            ),
            temporal_encoder=cls.TemporalEncoder(),
            forward_dynamics=cls.ForwardDynamics(
                dim=1024,
            ),
            policy=cls.Policy(
                dim=1024,
            ),
        )

    @classmethod
    def create_small(cls) -> Self:
        """Create small model configuration.

        VRAM usage is ~4GiB. (bfloat16)
        """
        return cls(
            image_jepa=cls.ImageJEPA(
                hidden_dim=320,
                depth=4,
                num_heads=2,
            ),
            audio_jepa=cls.AudioJEPA(
                hidden_dim=256,
                depth=4,
                num_heads=2,
            ),
            temporal_encoder=cls.TemporalEncoder(
                dim=768,
                depth=4,
            ),
            forward_dynamics=cls.ForwardDynamics(
                dim=768,
                depth=6,
            ),
            policy=cls.Policy(
                dim=768,
                depth=6,
            ),
        )

    @classmethod
    def create_tiny(cls) -> Self:
        """Create tiny model configuration.

        VRAM usage is ~2.5GiB. (bfloat16)
        """
        return cls(
            image_jepa=cls.ImageJEPA(
                hidden_dim=256,
                depth=2,
                num_heads=2,
            ),
            audio_jepa=cls.AudioJEPA(
                hidden_dim=192,
                depth=2,
                num_heads=2,
            ),
            temporal_encoder=cls.TemporalEncoder(
                dim=512,
                depth=2,
            ),
            forward_dynamics=cls.ForwardDynamics(
                dim=512,
                depth=4,
            ),
            policy=cls.Policy(
                dim=512,
                depth=4,
            ),
        )


# ======================================================================================
#                                TRAINER HYPERPARAMETERS
# ======================================================================================
# Training parameters control the learning dynamics, batch sizes, and optimization schedules.
# These are carefully tuned to balance with computational cost.


class TrainerHParams:
    """Trainer hyperparameter namespace controlling learning dynamics and
    optimization."""

    # fmt: off
    class ImageJEPA:
        lr: float = 0.0001                          # Conservative learning rate for stability
        batch_size: int = 32                        # Balance between gradient quality and memory
        min_new_data_count: int = 128               # Ensure sufficient fresh data before training
        mask_scale: tuple[float, float] = (0.025, 0.125)  # 2.5%-12.5% of patches masked per mask.
        num_masks: int = 4                          # Number of masks for multi-block masking.
        min_unmask_keep: int = 7                    # Minimum unmasked patches for context
        iteration_count: int = 16                   # Training iterations per trigger

    class AudioJEPA:
        lr: float = 0.0001                          # Conservative learning rate for stability
        batch_size: int = 32                        # Balance between gradient quality and memory
        min_new_data_count: int = 128               # Ensure sufficient fresh data before training
        mask_scale: tuple[float, float] = (0.1, 0.25)  # 10%-25% of patches masked per mask.
        num_masks: int = 4                          # Number of masks for multi-block masking.
        min_unmask_keep: int = 5                    # Minimum unmasked patches for context
        iteration_count: int = 16                   # Training iterations per trigger

    class TemporalEncoder:
        lr: float = 0.0001                          # Conservative for stable sequence learning
        seq_len: int = 32                           # Learning sequence length
        # Iteration count is max_samples / batch_size
        max_samples: int = 256                      # Total samples per training session
        batch_size: int = 8                         # Smaller batches due to sequence length
        min_new_data_count: int = 128               # Ensure sufficient fresh data before training

    class ForwardDynamics:
        lr: float = 0.0001                          # Learning rate.
        seq_len: int = 256                          # Long sequences for dynamics learning
        # Iteration count is max_samples / batch_size
        max_samples: int = 32                       # Iteration count
        batch_size: int = 1                         # Single sequences for memory efficiency
        min_new_data_count: int = 128               # Ensure sufficient fresh data before training

    class PPOPolicy:
        lr: float = 0.0001                          # Learning rate.
        seq_len: int = 256                          #  Long sequences consistent with forward dynamics
        # Iteration count is max_samples / batch_size
        max_samples: int = 32                       # Iteration count
        batch_size: int = 1                         # Single sequences for memory efficiency
        min_new_data_count: int = 128               # Ensure sufficient fresh data before training

    # fmt: on


# ======================================================================================
#                             DATA BUFFER HYPERPARAMETERS
# ======================================================================================
# Buffer sizes determine how much experience data is retained for training.
# These are calculated based on trainer requirements and expected data generation rates.


class DataBufferHParams:
    """Data buffer hyperparameter namespace controlling experience storage and
    retention."""

    class Image:
        max_size: int = (
            TrainerHParams.ImageJEPA.batch_size
            * TrainerHParams.AudioJEPA.iteration_count
        )  # Adjusting to training iteration count.

    class Audio:
        max_size: int = (
            TrainerHParams.AudioJEPA.batch_size
            * TrainerHParams.AudioJEPA.iteration_count
        )  # Adjusting to training iteration count.

    class Temporal:
        max_size = 1000  # ~100 seconds

    class ForwardDynamics:
        max_size = 1000  # ~100 seconds

    class Policy:
        max_size = 1000  # ~100 seconds


# ======================================================================================
#                                  LAUNCH ARGUMENTS
# ======================================================================================
# Command-line interface for configuring system deployment and resource allocation.


@dataclass
class CliArgs:
    """Command-line arguments for system configuration and deployment.

    These arguments allow flexible deployment across different hardware
    configurations and experimental setups without modifying code. The
    model size selection enables scaling from tiny (consumer pc) to
    large (workstation) configurations.
    """

    model_size: Literal["tiny", "small", "medium", "large", "huge"] = "tiny"
    """Model size selection."""

    device: str = "cuda"
    """Compute device for model."""

    precision: Literal["32", "bf16", "tf32"] = "bf16"
    """Computing precision."""

    output_dir: Path = PROJECT_ROOT / "logs"
    """Root directory to store states and logs."""


# ======================================================================================
#                                MAIN TRAINING PIPELINE
# ======================================================================================
# The main function orchestrates the complete training system, from component initialization
# to continuous learning execution. Each section builds upon previous components in a
# carefully designed dependency hierarchy.

import logging
import logging.handlers
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

logger = logging.getLogger(__name__)


def main() -> None:
    """Main training pipeline orchestrating the complete autonomous agent
    learning system.

    This function implements a carefully designed initialization sequence that builds
    complex learning components from foundational elements. The order of operations
    is critical - components have dependencies that must be satisfied before they
    can be properly configured.

    Pipeline Overview:
    1. System Configuration: Parse arguments and configure computational resources
    2. Logging Infrastructure: Set up comprehensive logging for debugging and monitoring
    3. Hyperparameter Resolution: Select appropriate model configuration for target hardware
    4. Component Creation: Build learning system components in dependency order
    5. Experiment Tracking: Initialize MLflow for experiment management and reproducibility
    6. System Launch: Start the continuous learning process with state persistence
    """

    # ==================================================================================
    #                              SYSTEM CONFIGURATION
    # ==================================================================================
    args = tyro.cli(CliArgs)

    device = torch.device(args.device)

    dtype = torch.float
    match args.precision:
        case "32":
            pass
        case "bf16":
            dtype = torch.bfloat16
        case "tf32":
            # Enable optimized matrix operations for improved training performance
            # This setting uses TensorFloat-32 (TF32) on Ampere GPUs for faster computation
            torch.set_float32_matmul_precision("medium")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================================
    #                              LOGGING INFRASTRUCTURE
    # ==================================================================================
    stream_handler = colorlog.StreamHandler()
    stream_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(blue)s%(asctime)s %(log_color)s%(levelname)s %(cyan)s[%(name)s] %(reset)s%(message)s",
        )
    )
    file_handler = logging.handlers.TimedRotatingFileHandler(
        args.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"),
        when="D",  # Daily rotation
        backupCount=6,  # 7 days logs
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])

    # ==================================================================================
    #                          HYPERPARAMETER RESOLUTION
    # ==================================================================================
    match args.model_size:
        case "huge":
            model_hparams = ModelHParams.create_huge()
        case "large":
            model_hparams = ModelHParams.create_large()
        case "medium":
            model_hparams = ModelHParams.create_medium()
        case "small":
            model_hparams = ModelHParams.create_small()
        case "tiny":
            model_hparams = ModelHParams.create_tiny()

    # ==================================================================================
    #                         NTERACTION SYSTEM CREATION
    # ==================================================================================
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

        # ======================================================
        #                 AGENT ARCHITECTURE
        # ======================================================
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
                    dtype=dtype,
                )
            ),
            curiosity_agent=CuriosityAgent(
                initial_forward_dynamics_hidden=torch.zeros(
                    model_hparams.forward_dynamics.depth,
                    model_hparams.forward_dynamics.dim,
                    device=device,
                    dtype=dtype,
                ),
                initial_policy_hidden=torch.zeros(
                    model_hparams.policy.depth,
                    model_hparams.policy.dim,
                    device=device,
                    dtype=dtype,
                ),
                max_imagination_steps=InteractionHParams.Agent.imagination_length,
                reward_average_method=average_exponentially,
                log_every_n_steps=round(1 / InteractionHParams.frame_interval),  # 1 sec
            ),
        )

        # ======================================================
        #              ENVIRONMENT CONFIGURATION
        # ======================================================
        hparams = InteractionHParams.Env
        environment = ModularEnvironment(
            sensor=SensorsDict(
                {
                    ObservationType.IMAGE: SensorWrapper(
                        sensors.ImageSensor(),
                        transforms.image.create_transform(
                            hparams.Obs.Image.size,
                            device=device,
                            dtype=dtype,
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
                            device=device,
                            dtype=dtype,
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
                transforms.action.ActionTransform(
                    hparams.Action.Mouse.max_velocity,
                    hparams.Action.Mouse.max_velocity,
                ),
            ),
        )
        interaction = FixedIntervalInteraction.with_sleep_adjustor(
            agent, environment, InteractionHParams.frame_interval
        )
        logger.info("Initialized Interaction Components.")
        return interaction

    # ==================================================================================
    #                         MODEL CREATION AND CONFIGURATION
    # ==================================================================================
    def create_models() -> dict[str, TrainingModel[Any]]:
        from pamiq_core.torch import TorchTrainingModel

        from sample.models import (
            ForwardDynamics,
            PolicyValueCommon,
            TemporalEncoder,
            create_audio_jepa,
            create_image_jepa,
        )
        from sample.models.temporal_encoder import ObsInfo as TemporalEncoderObsInfo
        from sample.transforms.action import ACTION_CHOICES

        temporal_encoder_obs_infos: dict[str, TemporalEncoderObsInfo] = {}

        # ======================================================
        #                  IMAGE JEPA MODELS
        # ======================================================
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
            dtype=dtype,
        )

        image_jepa_target_encoder = TorchTrainingModel(
            target_encoder,
            has_inference_model=True,
            inference_procedure=infer,
            device=device,
            dtype=dtype,
        )
        image_jepa_predictor = TorchTrainingModel(
            predictor,
            has_inference_model=False,
            device=device,
            dtype=dtype,
        )

        temporal_encoder_obs_infos[ObservationType.IMAGE] = TemporalEncoderObsInfo(
            dim=hparams.embed_dim,
            dim_hidden=ModelHParams.TemporalEncoder.image_dim,
            num_tokens=infer.output_patch_count,
        )

        # ======================================================
        #                  AUDIO JEPA MODELS
        # ======================================================
        hparams = model_hparams.audio_jepa
        context_encoder, target_encoder, predictor, infer = create_audio_jepa(
            sample_size=InteractionHParams.Env.Obs.Audio.frame_size,
            in_channels=InteractionHParams.Env.Obs.Audio.channel_size,
            hidden_dim=hparams.hidden_dim,
            embed_dim=hparams.embed_dim,
            depth=hparams.depth,
            num_heads=hparams.num_heads,
            output_downsample=hparams.output_downsample,
        )

        audio_jepa_context_encoder = TorchTrainingModel(
            context_encoder,
            has_inference_model=False,
            device=device,
            dtype=dtype,
        )

        audio_jepa_target_encoder = TorchTrainingModel(
            target_encoder,
            has_inference_model=True,
            inference_procedure=infer,
            device=device,
            dtype=dtype,
        )
        audio_jepa_predictor = TorchTrainingModel(
            predictor,
            has_inference_model=False,
            device=device,
            dtype=dtype,
        )

        temporal_encoder_obs_infos[ObservationType.AUDIO] = TemporalEncoderObsInfo(
            dim=hparams.embed_dim,
            dim_hidden=model_hparams.temporal_encoder.audio_dim,
            num_tokens=infer.output_patch_count,
        )

        # ======================================================
        #                TEMPORAL ENCODER MODEL
        # ======================================================
        hparams = model_hparams.temporal_encoder
        temporal_encoder = TorchTrainingModel(
            TemporalEncoder(
                obs_infos=temporal_encoder_obs_infos,
                dim=hparams.dim,
                depth=hparams.depth,
                dim_ff_hidden=hparams.dim * hparams.dim_ff_expansion_factor,
                dropout=hparams.dropout,
            ),
            has_inference_model=True,
            device=device,
            dtype=dtype,
            inference_procedure=TemporalEncoder.infer,
        )

        # ======================================================
        #                 FORWARD DYNAMICS MODEL
        # ======================================================
        hparams = model_hparams.forward_dynamics
        forward_dynamics = TorchTrainingModel(
            ForwardDynamics(
                obs_dim=model_hparams.temporal_encoder.dim,
                action_choices=list(ACTION_CHOICES),
                action_dim=hparams.action_dim,
                dim=hparams.dim,
                depth=hparams.depth,
                dim_ff_hidden=hparams.dim * hparams.dim_ff_expansion_factor,
                dropout=hparams.dropout,
            ),
            has_inference_model=True,
            device=device,
            dtype=dtype,
        )

        # ======================================================
        #                     POLICY MODEL
        # ======================================================
        hparams = model_hparams.policy
        policy = TorchTrainingModel(
            PolicyValueCommon(
                obs_dim=model_hparams.temporal_encoder.dim,
                action_choices=list(ACTION_CHOICES),
                dim=hparams.dim,
                depth=hparams.depth,
                dim_ff_hidden=hparams.dim * hparams.dim_ff_expansion_factor,
                dropout=hparams.dropout,
            ),
            device=device,
            dtype=dtype,
        )

        logger.info("Initialized Models.")
        return {
            ModelName.IMAGE_JEPA_CONTEXT_ENCODER: image_jepa_context_encoder,
            ModelName.IMAGE_JEPA_TARGET_ENCODER: image_jepa_target_encoder,
            ModelName.IMAGE_JEPA_PREDICTOR: image_jepa_predictor,
            ModelName.AUDIO_JEPA_CONTEXT_ENCODER: audio_jepa_context_encoder,
            ModelName.AUDIO_JEPA_TARGET_ENCODER: audio_jepa_target_encoder,
            ModelName.AUDIO_JEPA_PREDICTOR: audio_jepa_predictor,
            ModelName.TEMPORAL_ENCODER: temporal_encoder,
            ModelName.FORWARD_DYNAMICS: forward_dynamics,
            ModelName.POLICY_VALUE: policy,
        }

    # ==================================================================================
    #                        TRAINER CREATION AND CONFIGURATION
    # ==================================================================================
    def create_trainers() -> dict[str, Trainer]:
        from functools import partial

        from torch.optim import AdamW

        from sample.models.components.audio_patchifier import AudioPatchifier
        from sample.models.components.image_patchifier import ImagePatchifier
        from sample.trainers import (
            ImaginingForwardDynamicsTrainer,
            PPOPolicyTrainer,
            TemporalEncoderTrainer,
            jepa,
        )

        # ======================================================
        #                  IMAGE JEPA TRAINER
        # ======================================================
        hparams = TrainerHParams.ImageJEPA
        image_jepa = jepa.JEPATrainer(
            partial_optimizer=partial(AdamW, lr=hparams.lr),
            batch_size=hparams.batch_size,
            min_new_data_count=hparams.min_new_data_count,
            **jepa.IMAGE_CONFIG,
            collate_fn=jepa.MultiBlockMaskCollator2d(
                num_patches=ImagePatchifier.compute_num_patches(
                    InteractionHParams.Env.Obs.Image.size,
                    model_hparams.image_jepa.patch_size,
                ),
                mask_scale=hparams.mask_scale,
                n_masks=hparams.num_masks,
                min_keep=hparams.min_unmask_keep,
            ),
        )

        # ======================================================
        #                  AUDIO JEPA TRAINER
        # ======================================================
        hparams = TrainerHParams.AudioJEPA
        audio_jepa = jepa.JEPATrainer(
            partial_optimizer=partial(AdamW, lr=hparams.lr),
            batch_size=hparams.batch_size,
            min_new_data_count=hparams.min_new_data_count,
            **jepa.AUDIO_CONFIG,
            collate_fn=jepa.MultiBlockMaskCollator1d(
                num_patches=AudioPatchifier.compute_num_patches(
                    InteractionHParams.Env.Obs.Audio.frame_size
                ),
                mask_scale=hparams.mask_scale,
                n_masks=hparams.num_masks,
                min_keep=hparams.min_unmask_keep,
            ),
        )

        # ======================================================
        #                TEMPORAL ENCODER TRAINER
        # ======================================================
        hparams = TrainerHParams.TemporalEncoder
        temporal_encoder = TemporalEncoderTrainer(
            partial_optimzier=partial(AdamW, lr=hparams.lr),
            seq_len=hparams.seq_len,
            max_samples=hparams.max_samples,
            batch_size=hparams.batch_size,
            min_new_data_count=hparams.min_new_data_count,
            min_buffer_size=hparams.seq_len + 1,
        )

        # ======================================================
        #                FORWARD DYNAMICS TRAINER
        # ======================================================
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

        # ======================================================
        #                PPO POLICY TRAINER
        # ======================================================
        hparams = TrainerHParams.PPOPolicy
        policy = PPOPolicyTrainer(
            partial_optimizer=partial(AdamW, lr=hparams.lr),
            seq_len=hparams.seq_len,
            max_samples=hparams.max_samples,
            batch_size=hparams.batch_size,
            min_buffer_size=hparams.seq_len,
            min_new_data_count=hparams.min_new_data_count,
        )

        logger.info("Initialized Trainers.")
        return {
            "image_jepa": image_jepa,
            "audio_jepa": audio_jepa,
            "temporal_encoder": temporal_encoder,
            "forward_dynamics": forward_dynamics,
            "policy": policy,
        }

    # ==================================================================================
    #                              DATA BUFFER CREATION
    # ==================================================================================
    def create_data_buffers() -> dict[str, DataBuffer[Any]]:
        from pamiq_core.data.impls import RandomReplacementBuffer, SequentialBuffer

        from sample.data import DataKey

        image = RandomReplacementBuffer(
            collecting_data_names=[DataKey.OBSERVATION],
            max_size=DataBufferHParams.Image.max_size,
            expected_survival_length=int(
                # 12 hours of experience
                12 * 60 * 60 / InteractionHParams.frame_interval
            ),
        )

        audio = RandomReplacementBuffer(
            collecting_data_names=[DataKey.OBSERVATION],
            max_size=DataBufferHParams.Audio.max_size,
            expected_survival_length=int(
                # 12 hours of experience
                12 * 60 * 60 / InteractionHParams.frame_interval
            ),
        )

        temporal = SequentialBuffer(
            collecting_data_names=[DataKey.OBSERVATION, DataKey.HIDDEN],
            max_size=DataBufferHParams.Temporal.max_size,
        )

        forward_dynamics = SequentialBuffer(
            collecting_data_names=[DataKey.OBSERVATION, DataKey.ACTION, DataKey.HIDDEN],
            max_size=DataBufferHParams.ForwardDynamics.max_size,
        )

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

        logger.info("Initialized DataBuffers.")
        return {
            BufferName.IMAGE: image,
            BufferName.AUDIO: audio,
            BufferName.TEMPORAL: temporal,
            BufferName.FORWARD_DYNAMICS: forward_dynamics,
            BufferName.POLICY: policy,
        }

    # ==================================================================================
    #                      EXPERIMENT TRACKING AND SYSTEM LAUNCH
    # ==================================================================================

    mlflow.set_tracking_uri(args.output_dir / "mlflow")

    with mlflow.start_run():
        launch(
            interaction=create_interaction(),
            models=create_models(),
            data=create_data_buffers(),
            trainers=create_trainers(),
            config=LaunchConfig(
                states_dir=args.output_dir / "states",
                save_state_interval=24 * 60 * 60,  # Daily checkpoints (24 hours)
                max_keep_states=3,  # Retain 3 most recent checkpoints
                web_api_address=("0.0.0.0", 8391),  # Web API Address for control.
            ),
        )


if __name__ == "__main__":
    """Entry point for the VRChat autonomous agent training system.

    Example usage:
        python main.py --model_size large --device cuda --output_dir ./experiments/run_001
    """
    main()
