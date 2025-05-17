"""Tests for the image_sensor module."""

import numpy as np
import pytest

from pamiq_vrchat.sensors.image import ImageSensor


class TestImageSensor:
    """Tests for the ImageSensor class."""

    @pytest.fixture
    def mock_opencv_video_input(self, mocker):
        """Create a mock for OpenCVVideoInput."""
        mock = mocker.patch("pamiq_vrchat.sensors.image.OpenCVVideoInput")
        mock_instance = mock.return_value
        # Mock read method to return a simple frame
        mock_instance.read.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        return mock

    @pytest.fixture
    def mock_get_obs_camera_index(self, mocker):
        """Mock the get_obs_virtual_camera_index function."""
        return mocker.patch(
            "pamiq_vrchat.sensors.image.get_obs_virtual_camera_index", return_value=2
        )

    def test_init_with_camera_index(self, mock_opencv_video_input):
        """Test initialization with explicit camera index."""
        camera_index = 1
        ImageSensor(camera_index=camera_index)

        # Verify OpenCVVideoInput was called with the correct camera index
        mock_opencv_video_input.assert_called_once_with(camera_index)

    def test_init_without_camera_index(
        self, mock_opencv_video_input, mock_get_obs_camera_index
    ):
        """Test initialization without camera index (should use OBS virtual
        camera)."""
        ImageSensor()

        # Verify get_obs_virtual_camera_index was called
        mock_get_obs_camera_index.assert_called_once()

        # Verify OpenCVVideoInput was called with the index from get_obs_virtual_camera_index
        mock_opencv_video_input.assert_called_once_with(2)

    def test_read(self, mock_opencv_video_input):
        """Test the read method returns a frame."""
        sensor = ImageSensor(camera_index=0)
        frame = sensor.read()

        # Verify read was called
        mock_opencv_video_input.return_value.read.assert_called_once()

        # Verify frame has the expected shape and type
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (10, 10, 3)
        assert frame.dtype == np.uint8
