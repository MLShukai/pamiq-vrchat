import torch

from sample.models.components.fc_scalar_head import FCScalarHead


class TestFCScalarHead:
    def test_forward_no_squeeze(self):
        """Test that output shape is correct when squeeze_scalar_dim is
        False."""
        dim_in = 64
        batch_size = 32
        head = FCScalarHead(dim_in, squeeze_scalar_dim=False)

        x = torch.randn(batch_size, dim_in)
        out = head(x)

        assert out.shape == (batch_size, 1)

    def test_forward_with_squeeze(self):
        """Test that output shape is correct when squeeze_scalar_dim is
        True."""
        dim_in = 64
        batch_size = 32
        head = FCScalarHead(dim_in, squeeze_scalar_dim=True)

        x = torch.randn(batch_size, dim_in)
        out = head(x)

        assert out.shape == (batch_size,)

    def test_multi_dimensional_input(self):
        """Test that the module handles multi-dimensional inputs correctly."""
        dim_in = 64
        head = FCScalarHead(dim_in, squeeze_scalar_dim=False)

        # Test with 3D input
        x = torch.randn(8, 16, dim_in)
        out = head(x)
        assert out.shape == (8, 16, 1)

        # Test with 4D input
        x = torch.randn(4, 8, 12, dim_in)
        out = head(x)
        assert out.shape == (4, 8, 12, 1)

    def test_multi_dimensional_input_with_squeeze(self):
        """Test multi-dimensional inputs with squeeze_scalar_dim=True."""
        dim_in = 64
        head = FCScalarHead(dim_in, squeeze_scalar_dim=True)

        # Test with 3D input
        x = torch.randn(8, 16, dim_in)
        out = head(x)
        assert out.shape == (8, 16)

        # Test with 4D input
        x = torch.randn(4, 8, 12, dim_in)
        out = head(x)
        assert out.shape == (4, 8, 12)
