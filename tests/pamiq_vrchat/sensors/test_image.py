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

    def test_init_without_camera_index(
        self, mock_opencv_video_input, mock_get_obs_camera_index
    ):
        """Test initialization without camera index (should use OBS virtual
        camera)."""
        ImageSensor()

        # Verify get_obs_virtual_camera_index was called
        mock_get_obs_camera_index.assert_called_once()

        # Verify OpenCVVideoInput was called with the index from get_obs_virtual_camera_index
        assert mock_opencv_video_input.call_args[0][0] == 2

    def test_init_with_resolution(self, mock_opencv_video_input):
        """Test initialization with explicit width and height."""
        camera_index = 1
        width = 1920
        height = 1080
        ImageSensor(camera_index=camera_index, width=width, height=height)

        # Verify OpenCVVideoInput was called with the correct parameters
        mock_opencv_video_input.assert_called_once_with(camera_index, width, height)

    @pytest.mark.parametrize(
        "platform,expected_width,expected_height",
        [
            ("win32", 1280, 720),  # Windows should use default values
            ("linux", None, None),  # Non-Windows should not use default values
        ],
    )
    def test_init_platform_specific_defaults(
        self, mock_opencv_video_input, mocker, platform, expected_width, expected_height
    ):
        """Test platform-specific default resolution values."""
        # Mock sys.platform
        mocker.patch("sys.platform", platform)

        camera_index = 1
        ImageSensor(camera_index=camera_index)

        # Verify OpenCVVideoInput was called with the correct parameters
        mock_opencv_video_input.assert_called_once_with(
            camera_index, expected_width, expected_height
        )

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
