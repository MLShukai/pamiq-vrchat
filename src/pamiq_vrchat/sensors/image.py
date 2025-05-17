"""Image sensor module for VRChat interaction.

This module provides an image sensor implementation that captures frames
from a camera device, with specific support for OBS virtual camera.
The sensor can be used to obtain visual information from the VRChat environment.

Examples:
    >>> # Using default OBS virtual camera
    >>> sensor = ImageSensor()
    >>> frame = sensor.read()
    >>>
    >>> # Specifying a camera index
    >>> sensor = ImageSensor(camera_index=0)
    >>> frame = sensor.read()
"""

from typing import override

import cv2
import numpy as np
import numpy.typing as npt
from pamiq_core.interaction.modular_env import Sensor
from pamiq_io.video import OpenCVVideoInput

type ImageFrame = npt.NDArray[np.uint8]


class ImageSensor(Sensor[ImageFrame]):
    """Image sensor for capturing frames from a camera device.

    This class implements the Sensor interface for video input, allowing
    frames to be captured from a physical or virtual camera. It is
    designed to work with OBS virtual camera by default, but can be
    configured to use any available camera.
    """

    @override
    def __init__(self, camera_index: int | None = None) -> None:
        """Initializes an ImageSensor instance.

        Args:
            camera_index: Index of the camera to use. If None, automatically
                attempts to find and use the OBS virtual camera.
        """
        super().__init__()

        if camera_index is None:
            camera_index = get_obs_virtual_camera_index()

        self._input = OpenCVVideoInput(camera_index)

    @override
    def read(self) -> ImageFrame:
        """Reads a frame from the camera.

        Returns:
            A numpy array containing the image frame with shape (height, width, channels).

        Raises:
            RuntimeError: If a frame cannot be read from the camera.
        """
        return self._input.read()


def get_obs_virtual_camera_index() -> int:
    """Find the device index for OBS virtual camera.

    Returns:
        The device index of the OBS virtual camera.

    Raises:
        RuntimeError: If OBS virtual camera is not found.
    """
    from cv2_enumerate_cameras import enumerate_cameras, supported_backends

    for backend in supported_backends + (cv2.CAP_ANY,):
        for cam in enumerate_cameras(backend):
            if cam.name.strip() == "OBS Virtual Camera":
                return cam.index

    raise RuntimeError("Can not find OBS virtual camera device")
