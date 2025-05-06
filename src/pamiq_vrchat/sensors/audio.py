import re
from typing import override

import numpy as np
import numpy.typing as npt
import pulsectl
from pamiq_core.interaction.modular_env import Sensor
from pamiq_io.audio import SoundcardAudioInput

type AudioFrame = npt.NDArrray[np.float32]


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
        super().__init__()
        self._frame_size = frame_size
        if device_id is None:
            device_id = get_device_index_vrc_is_outputting_to()
        self._input = SoundcardAudioInput(sample_rate, device_id, block_size, channels)

    @override
    def read(self) -> AudioFrame:
        """Reads a frame from the Soundcard.

        Returns:
            A numpy array containing the image frame with shape (self._frame_size, channels).
        """
        return self._input.read(self._frame_size)


def get_device_index_vrc_is_outputting_to() -> int:
    """Find the speaker device VRChat.exe is outputting to.

    Returns:
        The device index VRChat.exe is using.

    Raises:
        RuntimeError: Speaker device VRChat.exe is outputting to is not found.
    """
    with pulsectl.Pulse("pamiq-vrchat") as p:
        for src_out in p.source_output_list():
            if re.match("VRChat.exe", src_out.proplist["application.name"]):
                return int(src_out.source)
    raise RuntimeError("Can not find speaker device VRChat.exe is outputting to.")
