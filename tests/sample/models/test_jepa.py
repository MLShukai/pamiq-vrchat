import pytest
import torch
import torch.nn as nn

from sample.models.components.image_patchifier import ImagePatchifier
from sample.models.components.positional_embeddings import (
    get_1d_positional_embeddings,
    get_2d_positional_embeddings,
)
from sample.models.jepa import AveragePoolInfer, Encoder, Predictor, create_image_jepa


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
        patchifier = ImagePatchifier(patch_size, 3, hidden_dim)
        positional_encodings = get_2d_positional_embeddings(
            hidden_dim, (img_size // patch_size, img_size // patch_size)
        ).reshape(n_patches, hidden_dim)

        encoder = Encoder(
            patchifier=patchifier,
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
        patchifier = ImagePatchifier(patch_size, 3, 64)
        positional_encodings = get_2d_positional_embeddings(
            64, (img_size // patch_size, img_size // patch_size)
        ).reshape(n_patches, 64)

        encoder = Encoder(
            patchifier=patchifier,
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
        patchifier = ImagePatchifier(8, 3, 64)

        with pytest.raises(
            ValueError,
            match="positional_encodings channel dimension must be hidden_dim.",
        ):
            Encoder(
                patchifier=patchifier,
                positional_encodings=torch.zeros(64, 32),  # Wrong channel size
                hidden_dim=64,
                embed_dim=32,
                depth=1,
                num_heads=2,
            )

        with pytest.raises(ValueError, match="positional_encodings must be 2d tensor!"):
            Encoder(
                patchifier=patchifier,
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
        patchifier = ImagePatchifier(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        encoder = Encoder(
            patchifier=patchifier,
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
        patchifier = ImagePatchifier(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        encoder = Encoder(
            patchifier=patchifier,
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
        patchifier = ImagePatchifier(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        encoder = Encoder(
            patchifier=patchifier,
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
        patchifier = ImagePatchifier(patch_size, 3, hidden_dim)
        positional_encodings = get_2d_positional_embeddings(
            hidden_dim, (n_patches_h, n_patches_w)
        ).reshape(n_patches, hidden_dim)

        encoder = Encoder(
            patchifier=patchifier,
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
    def encoder_1d(self):
        """Create encoder for 1D data."""
        n_patches = 16
        conv = nn.Conv1d(2, 64, kernel_size=8, stride=8)

        def patchifier(audio: torch.Tensor) -> torch.Tensor:
            out = conv(audio)
            return out.transpose(-1, -2)

        positional_encodings = get_1d_positional_embeddings(64, n_patches)

        return Encoder(
            patchifier=patchifier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=32,
            depth=1,
            num_heads=2,
        )

    @pytest.fixture
    def encoder_2d(self):
        """Create encoder for 2D data."""
        n_patches = 64
        patchifier = ImagePatchifier(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        return Encoder(
            patchifier=patchifier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=32,
            depth=1,
            num_heads=2,
        )

    @pytest.mark.parametrize(
        "ndim,num_patches,kernel_size,stride,error_msg",
        [
            (2, (4, 4, 4), 2, None, "Expected tuple of length 2, got 3"),
            (2, 16, (2, 2, 2), None, "Expected tuple of length 2, got 3"),
            (2, 16, 2, (1, 1, 1), "Expected tuple of length 2, got 3"),
        ],
    )
    def test_init_validation(self, ndim, num_patches, kernel_size, stride, error_msg):
        """Test initialization parameter validation."""
        with pytest.raises(ValueError, match=error_msg):
            AveragePoolInfer(
                ndim=ndim,
                num_patches=num_patches,
                kernel_size=kernel_size,
                stride=stride,
            )

    @pytest.mark.parametrize(
        "ndim,expected_pool_type",
        [
            (1, nn.AvgPool1d),
            (2, nn.AvgPool2d),
        ],
    )
    def test_pooling_layer_selection(self, ndim, expected_pool_type):
        """Test that correct pooling layer is selected."""
        pooler = AveragePoolInfer(ndim=ndim, num_patches=16, kernel_size=2)
        assert isinstance(pooler.pool, expected_pool_type)

    @pytest.mark.parametrize(
        "num_patches,expected",
        [
            (8, (8, 8)),
            ((8, 4), (8, 4)),
        ],
    )
    def test_validate_and_normalize(self, num_patches, expected):
        """Test parameter normalization."""
        pooler = AveragePoolInfer(ndim=2, num_patches=num_patches, kernel_size=2)
        assert pooler.num_patches == expected

    @pytest.mark.parametrize(
        "data_shape,expected_no_batch",
        [
            ((2, 128), (8, 32)),
            ((1, 2, 128), (1, 8, 32)),
            ((2, 3, 2, 128), (2, 3, 8, 32)),
        ],
    )
    def test_batch_handling_1d(self, encoder_1d, data_shape, expected_no_batch):
        """Test batch dimension handling for 1D data."""
        pooler = AveragePoolInfer(ndim=1, num_patches=16, kernel_size=2)
        audio = torch.randn(data_shape)
        result = pooler(encoder_1d, audio)
        assert result.shape == expected_no_batch

    @pytest.mark.parametrize(
        "kernel_size,expected_patches",
        [
            (1, 64),  # No reduction
            (2, 16),  # 8x8 -> 4x4
            (4, 4),  # 8x8 -> 2x2
        ],
    )
    def test_different_kernel_sizes_2d(self, encoder_2d, kernel_size, expected_patches):
        """Test different kernel sizes for 2D."""
        pooler = AveragePoolInfer(ndim=2, num_patches=(8, 8), kernel_size=kernel_size)
        image = torch.randn(1, 3, 64, 64)
        result = pooler(encoder_2d, image)
        assert result.shape == (1, expected_patches, 32)

    @pytest.mark.parametrize(
        "kernel_size,expected_patches",
        [
            (1, 16),  # No reduction
            (2, 8),  # 16 -> 8
            (4, 4),  # 16 -> 4
        ],
    )
    def test_different_kernel_sizes_1d(self, encoder_1d, kernel_size, expected_patches):
        """Test different kernel sizes for 1D."""
        pooler = AveragePoolInfer(ndim=1, num_patches=16, kernel_size=kernel_size)
        audio = torch.randn(1, 2, 128)
        result = pooler(encoder_1d, audio)
        assert result.shape == (1, expected_patches, 32)

    def test_custom_stride_1d(self, encoder_1d):
        """Test custom stride in 1D."""
        pooler = AveragePoolInfer(ndim=1, num_patches=16, kernel_size=3, stride=2)
        audio = torch.randn(1, 2, 128)
        result = pooler(encoder_1d, audio)
        assert result.shape == (1, 7, 32)  # (16-3)/2 + 1 = 7

    def test_asymmetric_2d_pooling(self, encoder_2d):
        """Test asymmetric 2D pooling parameters."""
        pooler = AveragePoolInfer(
            ndim=2, num_patches=(8, 8), kernel_size=(1, 2), stride=(1, 2)
        )
        image = torch.randn(1, 3, 64, 64)
        result = pooler(encoder_2d, image)
        assert result.shape == (1, 32, 32)  # 8x4 = 32


class TestCreateImageJEPA:
    @pytest.mark.parametrize(
        "image_size,patch_size,output_downsample",
        [
            (64, 8, 2),
            (224, 16, 4),
            ((96, 128), (12, 16), (2, 4)),
            (512, 32, 8),
        ],
    )
    def test_create_image_jepa_objects(
        self,
        image_size,
        patch_size,
        output_downsample,
    ):
        """Test that create_image_jepa creates objects of correct types."""
        (
            context_encoder,
            target_encoder,
            predictor,
            infer,
            num_patches,
        ) = create_image_jepa(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            hidden_dim=256,
            embed_dim=128,
            depth=2,
            num_heads=2,
            output_downsample=output_downsample,
        )

        # Check object types
        assert isinstance(context_encoder, Encoder)
        assert isinstance(target_encoder, Encoder)
        assert isinstance(predictor, Predictor)
        assert isinstance(infer, AveragePoolInfer)
        assert isinstance(num_patches, tuple)
        assert len(num_patches) == 2
        assert all(isinstance(p, int) for p in num_patches)

        # Check that target encoder is a separate instance (cloned)
        assert context_encoder is not target_encoder

        # Check that parameters are initially identical (cloned properly)
        for ctx_param, tgt_param in zip(
            context_encoder.parameters(), target_encoder.parameters()
        ):
            assert torch.equal(ctx_param, tgt_param)

    @pytest.mark.parametrize(
        "image_size,patch_size,output_downsample",
        [
            (64, 8, 2),
            (224, 16, 4),
            (96, 12, 2),
            ((128, 256), (16, 32), (2, 4)),
        ],
    )
    def test_patch_size_consistency(self, image_size, patch_size, output_downsample):
        """Test that final patch size matches actual encoder+infer output."""
        (
            context_encoder,
            _,
            _,
            infer,
            num_patches,
        ) = create_image_jepa(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            hidden_dim=128,
            embed_dim=32,
            depth=2,
            num_heads=2,
            output_downsample=output_downsample,
        )

        # Create test image
        if isinstance(image_size, tuple):
            height, width = image_size
        else:
            height = width = image_size

        test_image = torch.randn(1, 3, height, width)

        # Test actual forward pass through encoder + infer
        with torch.no_grad():
            output = infer(context_encoder, test_image)

        # Check that actual output matches expected patch dimensions
        expected_total_patches = num_patches[0] * num_patches[1]
        assert output.shape[1] == expected_total_patches
