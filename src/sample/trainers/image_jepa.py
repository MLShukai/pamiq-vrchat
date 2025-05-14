import itertools
import math
import random
from functools import partial
from multiprocessing import Value
from pathlib import Path
from typing import override

import mlflow
import torch
import torch.nn.functional as F
from pamiq_core import DataUser
from pamiq_core.torch import OptimizersSetup, TorchTrainer, get_device
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset, default_collate

from sample.data import BufferName, DataKey
from sample.models import ModelName
from sample.models.image_jepa import Encoder, Predictor
from sample.utils import size_2d, size_2d_to_int_tuple

OPTIMIZER_NAME = "optimizer"


class ImageJEPATrainer(TorchTrainer):
    """Trainer for Image Joint Embedding Predictive Architecture (I-JEPA).

    This trainer implements the JEPA training process which involves:
    1. A context encoder that encodes patches with some masked areas
    2. A target encoder that encodes full patches (without masking)
    3. A predictor that predicts target encoder outputs from context encoder outputs

    The training uses a masked prediction task where the model learns to predict
    representations of target patches from context patches. The target encoder
    is updated using an exponential moving average of the context encoder parameters.
    """

    @override
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[Tensor]],
        partial_optimizer: partial[Optimizer],
        target_encoder_update_moving_average: float = 0.996,  # based on the original I-JEPA initinal setting.
        max_epochs: int = 1,
        data_user_name: str = BufferName.IMAGE,
        min_buffer_size: int = 0,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize the ImageJEPA trainer.

        Args:
            partial_dataloader: Partially configured DataLoader to be used with
                dynamically created datasets during training.
            partial_optimizer: Partially configured optimizer to be used with
                the model parameters.
            target_encoder_update_moving_average: Momentum coefficient for updating
                the target encoder from the context encoder (higher values mean
                slower updates, default: 0.996 based on original I-JEPA).
            max_epochs: Maximum number of epochs to train per training session.
            data_user_name: Name of the data user providing training data.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
        """
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.data_user_name = data_user_name
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.target_encoder_update_moving_average = target_encoder_update_moving_average
        self.max_epochs = max_epochs
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
        self.image_data_user: DataUser[Tensor] = self.get_data_user(self.data_user_name)

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        training models are attached to the trainer. It retrieves and
        stores references to the JEPA models (context encoder, target
        encoder, and predictor) for convenient access during training.
        """
        super().on_training_models_attached()
        self.context_encoder = self.get_torch_training_model(
            ModelName.IMAGE_JEPA_CONTEXT_ENCODER, Encoder
        )
        self.target_encoder = self.get_torch_training_model(
            ModelName.IMAGE_JEPA_TARGET_ENCODER, Encoder
        )
        self.predictor = self.get_torch_training_model(
            ModelName.IMAGE_JEPA_PREDICTOR, Predictor
        )

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for JEPA training.

        Creates an optimizer that updates both the context encoder and predictor
        parameters. The target encoder is updated separately through exponential
        moving average and is not directly optimized.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {
            OPTIMIZER_NAME: self.partial_optimizer(
                itertools.chain(
                    self.context_encoder.model.parameters(),
                    self.predictor.model.parameters(),
                )
            )
        }

    @override
    def train(self) -> None:
        """Execute JEPA training process.

        This method implements the core JEPA training loop:
        1. Creates a dataset and dataloader from the collected images
        2. For each batch:
           - Encodes images with the target encoder (without gradients)
           - Encodes masked images with the context encoder
           - Uses the predictor to predict target patches from context
           - Computes smooth L1 loss between predictions and target representations
           - Updates context encoder and predictor parameters via backpropagation
           - Updates target encoder parameters via exponential moving average

        The target encoder serves as a momentum-updated teacher model that provides
        stable targets for the context encoder and predictor to learn from.
        """
        dataset = TensorDataset(
            torch.stack(list(self.image_data_user.get_data()[DataKey.OBSERVATION]))
        )
        dataloader = self.partial_dataloader(dataset=dataset)
        device = get_device(self.context_encoder.model)

        for _ in range(self.max_epochs):
            batch: tuple[Tensor, Tensor, Tensor]
            for batch in dataloader:
                (image_batch, masks_for_context_encoder, targets_for_predictor) = batch
                image_batch = image_batch.to(device)
                masks_for_context_encoder = masks_for_context_encoder.to(device)
                targets_for_predictor = targets_for_predictor.to(device)

                self.optimizers[OPTIMIZER_NAME].zero_grad()

                # target encoder
                with torch.no_grad():
                    latent_from_target_encoder: Tensor = self.target_encoder(
                        image_batch
                    )
                    # normalize over feature-dim
                    latent_from_target_encoder = F.layer_norm(
                        latent_from_target_encoder,
                        (latent_from_target_encoder.size(-1),),
                    )

                # context encoder
                latent_from_context_encoder = self.context_encoder(
                    image_batch, masks_for_context_encoder
                )

                # predictor
                latent_from_predictor = self.predictor(
                    latent_from_context_encoder, targets_for_predictor
                )

                # Element wise smooth l1 loss for masking.
                losses = F.smooth_l1_loss(
                    latent_from_predictor, latent_from_target_encoder, reduction="none"
                ).mean(-1)
                # shape: [batch, n_patches]

                # Ignore patches that are not selected for prediction.
                losses = torch.masked_fill(losses, ~targets_for_predictor, 0.0)
                loss = losses.sum() / targets_for_predictor.sum()

                loss.backward()
                self.optimizers[OPTIMIZER_NAME].step()

                # target_encoder updates weights by moving average from context_encoder
                with torch.no_grad():
                    # In the original I-JEPA, m changes through training process.
                    # But in ami-q, since assuming Semi-permanent training, m is set as fixed value.
                    m = self.target_encoder_update_moving_average
                    for target_encoder_param, context_encoder_param in zip(
                        self.target_encoder.model.parameters(),
                        self.context_encoder.model.parameters(),
                        strict=True,
                    ):
                        target_encoder_param.data.mul_(m).add_(
                            (1.0 - m) * context_encoder_param.detach().data
                        )
                metrics = {
                    "target_encoder_latent_std": latent_from_target_encoder.std(0)
                    .mean()
                    .item(),
                    "context_encoder_latent_std": latent_from_context_encoder.std(0)
                    .mean()
                    .item(),
                    "loss": loss.item(),
                }
                # logging
                mlflow.log_metrics(
                    {f"image-jepa/{tag}": v for tag, v in metrics.items()},
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


class MultiBlockMaskCollator:
    """I-JEPA collator function for providing boolean mask tensors.

    This collator creates boolean masks for both the context encoder and predictor target.
    It's designed to work with the I-JEPA (Image Joint Embedding Predictive Architecture) model.

    The masks are boolean tensors where:
    - True values indicate patches to be masked (ignored)
    - False values indicate patches to be processed or predicted

    This differs from IJEPAMaskCollator which uses integer indices for masked patches.
    """

    def __init__(
        self,
        input_size: size_2d,
        patch_size: size_2d,
        mask_scale: tuple[float, float] = (0.10, 0.25),
        n_masks: int = 4,
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        min_keep: int = 10,
    ) -> None:
        """Initialize the MultiBlockMaskCollator.

        Args:
            input_size: Size of the input image.
            patch_size: Size of each patch.
            mask_scale: Range of mask scale (min, max).
            n_masks: Number of mask candidates to generate.
            aspect_ratio: Range of aspect ratios for masks.
            min_keep: Minimum number of patches to keep unmasked.
        """
        super().__init__()
        if mask_scale[0] > mask_scale[1]:
            raise ValueError("mask_scale[0] must be less than mask_scale[1]")
        if mask_scale[0] < 0:
            raise ValueError("mask_scale[0] must be greater than 0")
        if mask_scale[1] > 1:
            raise ValueError("mask_scale[1] must be less than 1")

        input_size = size_2d_to_int_tuple(input_size)
        self.patch_size = size_2d_to_int_tuple(patch_size)

        if input_size[0] % self.patch_size[0] != 0:
            raise ValueError(
                f"Input height {input_size[0]} must be divisible by patch height {self.patch_size[0]}"
            )
        if input_size[1] % self.patch_size[1] != 0:
            raise ValueError(
                f"Input width {input_size[1]} must be divisible by patch width {self.patch_size[1]}"
            )

        self.n_patches_height = input_size[0] // self.patch_size[0]
        self.n_patches_width = input_size[1] // self.patch_size[1]

        if min_keep > self.n_patches:
            raise ValueError(
                f"min_keep ({min_keep}) must be less than or equal to total patches "
                f"({self.n_patches_height * self.n_patches_width})"
            )

        self.mask_scale = mask_scale
        self.n_masks = n_masks
        self.aspect_ratio = aspect_ratio
        self.min_keep = min_keep
        self._itr_counter = Value("i", random.randrange(2**32))

    @property
    def n_patches(self) -> int:
        """Total number of patches in the image grid."""
        return self.n_patches_height * self.n_patches_width

    def step(self) -> int:
        """Increment and return the iteration counter."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask_rectangle(
        self,
        generator: torch.Generator,
    ) -> tuple[int, int, int, int]:
        """Randomly sample a rectangular mask.

        Args:
            generator: Generator for pseudo-random numbers.

        Returns:
            Top, bottom, left, and right coordinates of the mask.
        """
        _scale_rand, _ratio_rand = torch.rand(2, generator=generator).tolist()
        # -- Sample mask scale
        min_s, max_s = self.mask_scale
        mask_scale = min_s + _scale_rand * (max_s - min_s)
        max_keep = mask_scale * self.n_patches

        # -- Sample mask aspect-ratio
        min_ar, max_ar = self.aspect_ratio
        aspect_ratio = min_ar + _ratio_rand * (max_ar - min_ar)

        # -- Compute height and width of mask (given scale and aspect-ratio)
        patch_ar = self.n_patches_width / self.n_patches_height
        if patch_ar > aspect_ratio:
            h_max = self.n_patches_height
            w_max = self.n_patches_height * aspect_ratio
        else:
            h_max = self.n_patches_width / aspect_ratio
            w_max = self.n_patches_width

        num_patches_max = h_max * w_max
        scale = math.sqrt(max_keep / num_patches_max)
        h, w = round(scale * h_max), round(scale * w_max)

        # Apply min keep
        if h * w < self.min_keep:
            scale = math.sqrt(self.min_keep / num_patches_max)
            h, w = math.ceil(scale * h_max), math.ceil(scale * w_max)

        # clamp
        h = min(max(h, 1), self.n_patches_height)
        w = min(max(w, 1), self.n_patches_width)

        # -- Compute mask coordinates
        top = int(
            torch.randint(
                high=self.n_patches_height - h + 1, size=(1,), generator=generator
            ).item()
        )
        left = int(
            torch.randint(
                high=self.n_patches_width - w + 1, size=(1,), generator=generator
            ).item()
        )
        bottom = top + h
        right = left + w

        return top, bottom, left, right

    def sample_masks_and_target(
        self, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        """Sample boolean masks for the encoder and a target mask for the
        predictor.

        Args:
            generator: Generator for pseudo-random numbers.

        Returns:
            A tuple containing:
                - encoder_mask: Boolean mask for the encoder (True for masked patches)
                - predictor_target: Boolean mask representing the target for the predictor
        """
        sampled_masks = []
        for _ in range(self.n_masks):
            mask = torch.zeros(
                self.n_patches_height, self.n_patches_width, dtype=torch.bool
            )
            top, bottom, left, right = self._sample_mask_rectangle(generator)
            mask[top:bottom, left:right] = True
            sampled_masks.append(mask.flatten())

        # Create encoder mask by combining all sampled masks
        encoder_mask = torch.stack(sampled_masks).sum(0).type(torch.bool)

        # Randomly select one mask as the predictor target
        mask_idx = int(
            torch.randint(
                high=len(sampled_masks), size=(1,), generator=generator
            ).item()
        )
        predictor_target = sampled_masks[mask_idx]

        # Apply min keep
        if encoder_mask.logical_not().sum() < self.min_keep:
            indices = torch.randperm(self.n_patches, generator=generator)[
                : self.min_keep
            ]
            encoder_mask[indices] = False
            predictor_target[indices] = True

        return encoder_mask, predictor_target

    def __call__(self, images: list[tuple[Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
        """Collate input images and create boolean masks for context encoder
        and predictor target.

        Args:
            images: List of image tensors. Each image is shape [3, height, width].

        Returns:
            A tuple containing:
                - collated_images: Collated images (shape: [batch_size, 3, height, width])
                - collated_encoder_masks: Boolean masks for context encoder (shape: [batch_size, n_patches])
                - collated_predictor_targets: Boolean masks representing predictor targets
                  (shape: [batch_size, n_patches])
        """
        collated_images: Tensor = default_collate(images)[0]

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        encoder_masks, predictor_targets = [], []
        for _ in range(len(images)):
            enc_mask, pred_target = self.sample_masks_and_target(g)
            encoder_masks.append(enc_mask)
            predictor_targets.append(pred_target)

        collated_encoder_masks = torch.stack(encoder_masks)
        collated_predictor_targets = torch.stack(predictor_targets)

        return (
            collated_images,
            collated_encoder_masks,
            collated_predictor_targets,
        )
