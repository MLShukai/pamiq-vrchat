import pytest
import torch

from sample.models.components.patch_embedding import PatchEmbedding
from sample.models.components.positional_embeddings import get_2d_positional_embeddings
from sample.models.jepa import AveragePoolInfer2d, Encoder, Predictor


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
        n_patches = (img_size // patch_size) ** 2
        patchfier = PatchEmbedding(patch_size, 3, hidden_dim)
        positional_encodings = get_2d_positional_embeddings(
            hidden_dim, (img_size // patch_size, img_size // patch_size)
        ).reshape(n_patches, hidden_dim)

        encoder = Encoder(
            patchfier=patchfier,
            positional_encodings=positional_encodings,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            depth=2,
            num_heads=2,
        )

        images = torch.randn(batch_size, 3, img_size, img_size)
        encoded = encoder(images)

        assert encoded.shape == (batch_size, n_patches, embed_dim)

    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("img_size", [64])
    @pytest.mark.parametrize("patch_size", [8])
    @pytest.mark.parametrize("mask_ratio", [0.25])
    def test_forward_with_mask(self, batch_size, img_size, patch_size, mask_ratio):
        """Test Encoder's forward pass with mask."""
        n_patches = (img_size // patch_size) ** 2
        patchfier = PatchEmbedding(patch_size, 3, 64)
        positional_encodings = get_2d_positional_embeddings(
            64, (img_size // patch_size, img_size // patch_size)
        ).reshape(n_patches, 64)

        encoder = Encoder(
            patchfier=patchfier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=32,
            depth=2,
            num_heads=2,
        )

        images = torch.randn(batch_size, 3, img_size, img_size)

        # Create a random mask with the specified ratio
        num_mask = int(n_patches * mask_ratio)
        masks = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        for i in range(batch_size):
            mask_indices = torch.randperm(n_patches)[:num_mask]
            masks[i, mask_indices] = True

        encoded = encoder(images, masks)

        assert encoded.shape == (batch_size, n_patches, encoder.out_proj.out_features)

    def test_invalid_positional_encoding_shape(self):
        """Test error when positional encoding shape doesn't match expected
        shape."""
        patchfier = PatchEmbedding(8, 3, 64)

        with pytest.raises(
            ValueError,
            match="positional_encodings channel dimension must be hidden_dim.",
        ):
            Encoder(
                patchfier=patchfier,
                positional_encodings=torch.zeros(64, 32),  # Wrong channel size
                hidden_dim=64,
                embed_dim=32,
                depth=1,
                num_heads=2,
            )

        with pytest.raises(ValueError, match="positional_encodings must be 2d tensor!"):
            Encoder(
                patchfier=patchfier,
                positional_encodings=torch.zeros(
                    64,
                ),  # Wrong dims size
                hidden_dim=64,
                embed_dim=32,
                depth=1,
                num_heads=2,
            )

    def test_invalid_mask_shape(self):
        """Test error when mask shape doesn't match encoded image shape."""
        n_patches = 64
        patchfier = PatchEmbedding(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        encoder = Encoder(
            patchfier=patchfier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=64,
            depth=1,
            num_heads=2,
        )
        images = torch.randn(2, 3, 64, 64)

        # Create mask with incorrect shape
        masks = torch.zeros(2, n_patches - 1, dtype=torch.bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            encoder(images, masks)

    def test_non_bool_mask(self):
        """Test error when mask tensor is not boolean."""
        n_patches = 64
        patchfier = PatchEmbedding(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        encoder = Encoder(
            patchfier=patchfier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=64,
            depth=1,
            num_heads=2,
        )
        images = torch.randn(1, 3, 64, 64)

        # Create mask with incorrect dtype (float instead of bool)
        masks = torch.zeros(1, n_patches, dtype=torch.float32)

        with pytest.raises(ValueError, match="Mask tensor dtype must be bool"):
            encoder(images, masks)

    def test_clone(self):
        n_patches = 64
        patchfier = PatchEmbedding(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        encoder = Encoder(
            patchfier=patchfier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
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
    @pytest.mark.parametrize("n_patches", [64])
    @pytest.mark.parametrize("embed_dim", [32])
    @pytest.mark.parametrize("hidden_dim", [32])
    def test_forward(self, batch_size, n_patches, embed_dim, hidden_dim):
        """Test Predictor's forward pass."""
        positional_encodings = get_2d_positional_embeddings(hidden_dim, (8, 8)).reshape(
            n_patches, hidden_dim
        )

        predictor = Predictor(
            positional_encodings=positional_encodings,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            depth=2,
            num_heads=2,
        )

        # Create latents as if they came from encoder
        latents = torch.randn(batch_size, n_patches, embed_dim)

        # Create target mask (e.g., 25% of patches are targets)
        targets = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        for i in range(batch_size):
            target_indices = torch.randperm(n_patches)[: n_patches // 4]
            targets[i, target_indices] = True

        predictions = predictor(latents, targets)

        # Check output shape
        assert predictions.shape == (
            batch_size,
            n_patches,
            embed_dim,
        )

    def test_invalid_positional_encoding_shape(self):
        """Test error when positional encoding shape doesn't match expected
        shape."""

        with pytest.raises(
            ValueError,
            match="positional_encodings channel dimension must be hidden_dim.",
        ):
            Predictor(
                positional_encodings=torch.zeros(
                    32, 64
                ),  # Wrong shape for hidden_dim=32
                embed_dim=32,
                hidden_dim=32,
                depth=1,
                num_heads=2,
            )

        with pytest.raises(ValueError, match="positional_encodings must be 2d tensor!"):
            Predictor(
                positional_encodings=torch.zeros(
                    32,
                ),  # Wrong dim size
                embed_dim=32,
                hidden_dim=32,
                depth=1,
                num_heads=2,
            )

    def test_invalid_target_shape(self):
        """Test error when target shape doesn't match latent shape."""
        n_patches = 64
        positional_encodings = get_2d_positional_embeddings(32, (8, 8)).reshape(
            n_patches, 32
        )

        predictor = Predictor(
            positional_encodings=positional_encodings,
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
        n_patches = 64
        positional_encodings = get_2d_positional_embeddings(32, (8, 8)).reshape(
            n_patches, 32
        )

        predictor = Predictor(
            positional_encodings=positional_encodings,
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


class TestJEPAIntegration:
    def test_encoder_predictor_integration(self):
        """Test that encoder and predictor work together in a typical
        workflow."""
        # Create encoder and predictor with smaller dimensions
        img_size = 64
        patch_size = 8
        embed_dim = 32
        hidden_dim = 64

        # Calculate grid dimensions
        n_patches_h = img_size // patch_size
        n_patches_w = img_size // patch_size
        n_patches = n_patches_h * n_patches_w

        # Initialize models with reduced complexity
        patchfier = PatchEmbedding(patch_size, 3, hidden_dim)
        positional_encodings = get_2d_positional_embeddings(
            hidden_dim, (n_patches_h, n_patches_w)
        ).reshape(n_patches, hidden_dim)

        encoder = Encoder(
            patchfier=patchfier,
            positional_encodings=positional_encodings,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            depth=2,
            num_heads=2,
        )

        predictor = Predictor(
            positional_encodings=positional_encodings[
                :, :32
            ],  # Use first 32 dims for predictor
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


class TestAveragePoolInfer:
    @pytest.fixture
    def encoder(self):
        """Create a minimal working Encoder for testing."""
        n_patches = 16
        patchfier = PatchEmbedding(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (4, 4)).reshape(
            n_patches, 64
        )

        return Encoder(
            patchfier=patchfier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=32,
            depth=1,
            num_heads=2,
        )

    def test_basic_functionality(self, encoder):
        """Test basic pooling functionality."""
        # Setup
        pooler = AveragePoolInfer2d(num_patches=(4, 4), kernel_size=2)
        image = torch.randn(1, 3, 32, 32)

        # Encode and pool
        result = pooler(encoder, image)

        # With 4x4 patches and kernel_size=2, we should get 2x2=4 patches
        assert result.shape == (1, 4, 32)

    def test_different_batch_sizes(self, encoder):
        """Test with different batch sizes."""
        pooler = AveragePoolInfer2d(num_patches=(4, 4), kernel_size=2)

        # Test with batch size 2
        image_batch2 = torch.randn(2, 3, 32, 32)
        result_batch2 = pooler(encoder, image_batch2)
        assert result_batch2.shape == (2, 4, 32)

        # Test with batch size 4
        image_batch4 = torch.randn(4, 3, 32, 32)
        result_batch4 = pooler(encoder, image_batch4)
        assert result_batch4.shape == (4, 4, 32)

    def test_no_batch_dimension(self, encoder):
        """Test with input tensor that has no batch dimension."""
        pooler = AveragePoolInfer2d(num_patches=(4, 4), kernel_size=2)

        # Create an image without batch dimension
        image = torch.randn(3, 32, 32)

        # Execute
        result = pooler(encoder, image)

        # Verify the result has no batch dimension
        assert result.shape == (4, 32)

    def test_multi_dimensional_batch(self, encoder):
        """Test with multi-dimensional batch."""
        pooler = AveragePoolInfer2d(num_patches=(4, 4), kernel_size=2)

        # Create a batch with multiple dimensions [2, 3, 3, 32, 32]
        image = torch.randn(2, 3, 3, 32, 32)

        # Execute
        result = pooler(encoder, image)

        # Verify the result maintains the batch dimensions
        assert result.shape == (2, 3, 4, 32)

    def test_different_kernel_sizes(self, encoder):
        """Test with different kernel sizes."""
        # Original number of patches is 4x4=16

        # Test with kernel size 1 (no reduction)
        pooler1 = AveragePoolInfer2d(num_patches=(4, 4), kernel_size=1)
        image = torch.randn(1, 3, 32, 32)
        result1 = pooler1(encoder, image)
        assert result1.shape == (1, 16, 32)  # No reduction

        # Test with kernel size (2, 1) (reduction only in height)
        pooler2 = AveragePoolInfer2d(num_patches=(4, 4), kernel_size=(2, 1))
        result2 = pooler2(encoder, image)
        assert result2.shape == (1, 8, 32)  # 4x4 -> 2x4 = 8 patches

    def test_custom_stride(self, encoder):
        """Test with custom stride value."""
        # Test with stride different from kernel size
        pooler = AveragePoolInfer2d(num_patches=(4, 4), kernel_size=2, stride=1)
        image = torch.randn(1, 3, 32, 32)
        result = pooler(encoder, image)

        # With stride 1, we should get (4-2+1) x (4-2+1) = 3x3 = 9 patches
        assert result.shape == (1, 9, 32)
