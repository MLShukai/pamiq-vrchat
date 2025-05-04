import pytest
import torch

from sample.models.image_jepa import Encoder, Predictor
from sample.utils import size_2d_to_int_tuple


class TestEncoder:
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("img_size", [64, 96])
    @pytest.mark.parametrize("patch_size", [8])
    @pytest.mark.parametrize("hidden_dim", [64])
    @pytest.mark.parametrize("embed_dim", [32])
    def test_forward_without_mask(
        self, batch_size, img_size, patch_size, hidden_dim, embed_dim
    ):
        """Test Encoder's forward pass without mask."""
        encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            depth=2,
            num_heads=2,
        )

        images = torch.randn(batch_size, 3, img_size, img_size)
        encoded = encoder(images)

        n_patches = (img_size // patch_size) ** 2
        assert encoded.shape == (batch_size, n_patches, embed_dim)

    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("img_size", [64])
    @pytest.mark.parametrize("patch_size", [8])
    @pytest.mark.parametrize("mask_ratio", [0.25])
    def test_forward_with_mask(self, batch_size, img_size, patch_size, mask_ratio):
        """Test Encoder's forward pass with mask."""
        encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            hidden_dim=64,
            embed_dim=32,
            depth=2,
            num_heads=2,
        )

        images = torch.randn(batch_size, 3, img_size, img_size)
        n_patches = (img_size // patch_size) ** 2

        # Create a random mask with the specified ratio
        num_mask = int(n_patches * mask_ratio)
        masks = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        for i in range(batch_size):
            mask_indices = torch.randperm(n_patches)[:num_mask]
            masks[i, mask_indices] = True

        encoded = encoder(images, masks)

        assert encoded.shape == (batch_size, n_patches, encoder.out_proj.out_features)

    @pytest.mark.parametrize(
        "img_size,patch_size,expected_error",
        [
            (63, 8, "Image height 63 must be divisible by patch height 8"),
            (64, 9, "Image height 64 must be divisible by patch height 9"),
        ],
    )
    def test_invalid_image_size(self, img_size, patch_size, expected_error):
        """Test error when image size is not divisible by patch size."""
        with pytest.raises(ValueError, match=expected_error):
            Encoder(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=64,
                depth=1,
                num_heads=2,
            )

    def test_invalid_mask_shape(self):
        """Test error when mask shape doesn't match encoded image shape."""
        encoder = Encoder(
            img_size=64,
            patch_size=8,
            embed_dim=64,
            depth=1,
            num_heads=2,
        )
        images = torch.randn(2, 3, 64, 64)
        n_patches = (64 // 8) ** 2

        # Create mask with incorrect shape
        masks = torch.zeros(2, n_patches - 1, dtype=torch.bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            encoder(images, masks)

    @pytest.mark.parametrize("img_size", [64])
    @pytest.mark.parametrize("patch_size", [8])
    def test_image_patch_size_variations(self, img_size, patch_size):
        """Test that encoder handles both int and tuple for img_size and
        patch_size."""
        encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=64,
            depth=1,
            num_heads=2,
        )

        # Regardless of the input format, we should be able to use the encoder
        img_size = size_2d_to_int_tuple(img_size)
        patch_size = size_2d_to_int_tuple(patch_size)
        h, w = img_size

        ph, pw = patch_size

        images = torch.randn(1, 3, h, w)
        encoded = encoder(images)

        n_patches = (h // ph) * (w // pw)
        assert encoded.shape == (1, n_patches, encoder.out_proj.out_features)

    def test_non_bool_mask(self):
        """Test error when mask tensor is not boolean."""
        encoder = Encoder(
            img_size=64,
            patch_size=8,
            embed_dim=64,
            depth=1,
            num_heads=2,
        )
        images = torch.randn(1, 3, 64, 64)
        n_patches = (64 // 8) ** 2

        # Create mask with incorrect dtype (float instead of bool)
        masks = torch.zeros(1, n_patches, dtype=torch.float32)

        with pytest.raises(ValueError, match="Mask tensor dtype must be bool"):
            encoder(images, masks)

        # Test with int dtype
        masks = torch.zeros(1, n_patches, dtype=torch.int32)

        with pytest.raises(ValueError, match="Mask tensor dtype must be bool"):
            encoder(images, masks)

    def test_clone(self):
        encoder = Encoder(
            img_size=64,
            patch_size=8,
            embed_dim=64,
            depth=1,
            num_heads=2,
        )
        copied = encoder.clone()
        assert encoder is not copied
        for p, p_copied in zip(encoder.parameters(), copied.parameters(), strict=True):
            assert torch.equal(p, p_copied)


class TestPredictor:
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("n_patches", [8, (8, 8)])
    @pytest.mark.parametrize("embed_dim", [32])
    @pytest.mark.parametrize("hidden_dim", [32])
    def test_forward(self, batch_size, n_patches, embed_dim, hidden_dim):
        """Test Predictor's forward pass."""
        predictor = Predictor(
            n_patches=n_patches,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            depth=2,
            num_heads=2,
        )

        # Determine actual number of patches
        n_patches = size_2d_to_int_tuple(n_patches)
        n_patches_count = n_patches[0] * n_patches[1]

        # Create latents as if they came from encoder
        latents = torch.randn(batch_size, n_patches_count, embed_dim)

        # Create target mask (e.g., 25% of patches are targets)
        targets = torch.zeros(batch_size, n_patches_count, dtype=torch.bool)
        for i in range(batch_size):
            target_indices = torch.randperm(n_patches_count)[: n_patches_count // 4]
            targets[i, target_indices] = True

        predictions = predictor(latents, targets)

        # Check output shape
        assert predictions.shape == (
            batch_size,
            n_patches_count,
            embed_dim,
        )

    def test_invalid_target_shape(self):
        """Test error when target shape doesn't match latent shape."""
        predictor = Predictor(
            n_patches=(8, 8),
            embed_dim=32,
            hidden_dim=32,
            depth=1,
            num_heads=2,
        )
        latents = torch.randn(1, 64, 32)
        targets = torch.zeros(1, 32, dtype=torch.bool)  # Incorrect shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            predictor(latents, targets)

    def test_non_bool_target(self):
        """Test error when target tensor is not boolean."""
        predictor = Predictor(
            n_patches=(8, 8),
            embed_dim=32,
            hidden_dim=32,
            depth=1,
            num_heads=2,
        )
        latents = torch.randn(1, 64, 32)

        # Create targets with incorrect dtype (float instead of bool)
        targets = torch.zeros(1, 64, dtype=torch.float32)

        with pytest.raises(ValueError, match="Target tensor dtype must be bool"):
            predictor(latents, targets)

        # Test with int dtype
        targets = torch.zeros(1, 64, dtype=torch.int32)

        with pytest.raises(ValueError, match="Target tensor dtype must be bool"):
            predictor(latents, targets)


class TestJEPAIntegration:
    def test_encoder_predictor_integration(self):
        """Test that encoder and predictor work together in a typical
        workflow."""
        # Create encoder and predictor with smaller dimensions
        img_size = 64
        patch_size = 8
        embed_dim = 32

        # Calculate grid dimensions
        img_size_tuple = size_2d_to_int_tuple(img_size)
        patch_size_tuple = size_2d_to_int_tuple(patch_size)
        n_patches_h = img_size_tuple[0] // patch_size_tuple[0]
        n_patches_w = img_size_tuple[1] // patch_size_tuple[1]
        n_patches = n_patches_h * n_patches_w

        # Initialize models with reduced complexity
        encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            hidden_dim=64,
            embed_dim=embed_dim,
            depth=2,
            num_heads=2,
        )

        predictor = Predictor(
            n_patches=(n_patches_h, n_patches_w),
            embed_dim=embed_dim,
            hidden_dim=32,
            depth=1,
            num_heads=2,
        )

        # Create a smaller batch of images
        batch_size = 1
        images = torch.randn(batch_size, 3, img_size, img_size)

        # Create context and target masks (non-overlapping)
        context_mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        target_mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        target_mask[:, n_patches // 2 :] = True

        # Encode with context mask
        encoded = encoder(images, context_mask)

        # Predict with target mask
        predictions = predictor(encoded, target_mask)

        # Check shapes
        assert encoded.shape == (batch_size, n_patches, embed_dim)
        assert predictions.shape == (batch_size, n_patches, embed_dim)
