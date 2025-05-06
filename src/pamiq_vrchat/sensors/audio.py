import logging
import re
import shutil
import subprocess
from typing import override

import numpy as np
import numpy.typing as npt
from pamiq_core.interaction.modular_env import Sensor

logger = logging.getLogger(__name__)

type AudioFrame = npt.NDArray[np.float32]


class AudioSensor(Sensor[AudioFrame]):
    """Capturing audio from a input device.

    The default input device is the one VRChat.exe is using, but can be
    configured by the argument.
    """

    def __init__(
        self,
        frame_size: int,
        sample_rate: int = 16000,
        channels: int = 2,
        device_id: str | None = None,
        block_size: int | None = None,
    ):
        """Initializes an AudioSensor instance.

        Args:
            frame_size:
                Number of samples the user needs.
            sample_rate:
                Sample rate.
            channels:
                Audio channels.
            device_id:
                Index of an audio device input to a model. If None, automatically tries to find the device used by VRChat.exe.
            block_size:
                 Number of samples SoundCard reads.
        """
        from pamiq_io.audio import SoundcardAudioInput

        super().__init__()
        self._frame_size = frame_size
        if device_id is None:
            device_id = get_device_id_vrchat_is_outputting_to()
        self._input = SoundcardAudioInput(sample_rate, device_id, block_size, channels)

    @override
    def read(self) -> AudioFrame:
        """Reads a frame from the Soundcard.

        Returns:
            A numpy array containing the image frame with shape (self._frame_size, channels).
        """
        return self._input.read(self._frame_size)


def get_device_id_vrchat_is_outputting_to() -> str | None:
    """Find the speaker device VRChat.exe is outputting to.

    Returns:
        The device index VRChat.exe is using.

    Raises:
        RuntimeError: Speaker device VRChat.exe is outputting to is not found.
    """
    if shutil.which("pactl") is None:
        logger.warning("pactl command is not found.")
        return

    pactl_output = subprocess.check_output(
        ["pactl", "list", "source-outputs"], text=True
    )

    vrchat_section = re.search(
        r'Source Output #\d+.*?application\.name = "VRChat\.exe".*?',
        pactl_output,
        re.DOTALL,
    )

    if not vrchat_section:
        return None

    source_match = re.search(r"Source: (\d+)", vrchat_section.group(0))
    if not source_match:
        return None

    source_id = source_match.group(1)

    sources_output = subprocess.check_output(
        ["pactl", "list", "sources", "short"], text=True
    )

    for line in sources_output.splitlines():
        parts = line.split()
        if parts and parts[0] == source_id:
            return parts[1]

    logger.warning("Can not find speaker device VRChat.exe is outputting to.")
    return None
