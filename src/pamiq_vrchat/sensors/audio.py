import re
from typing import override

import numpy as np
import numpy.typing as npt
import pulsectl
from pamiq_core.interaction.modular_env import Sensor
from pamiq_io.audio import SoundcardAudioInput

type AudioFrame = npt.NDArrray[np.float32]


class AudioSensor(Sensor[AudioFrame]):
    def __init__(
        self,
        frame_size: int,
        sample_rate: int = 16000,
        channels: int = 2,
        device_id: str | None = None,
        block_size: int | None = None,
    ):
        super().__init__()
        self._frame_size = frame_size
        if device_id is None:
            device_id = get_vrchat_audio_input_device_index()
        self._input = SoundcardAudioInput(sample_rate, device_id, block_size, channels)

    @override
    def read(self) -> AudioFrame:
        return self._input.read(self._frame_size)


def get_vrchat_audio_input_device_index() -> int:
    """Find the device index for OBS virtual camera.

    This function uses v4l2-ctl to list all video devices and find
    the OBS virtual camera among them.

    Returns:
        The device index of the OBS virtual camera.

    Raises:
        RuntimeError: If v4l2-ctl is not installed or if OBS virtual camera is not found.
    """
    with pulsectl.Pulse("pamiq-vrchat") as p:
        vrchat_source = None
        for src_out in p.source_output_list():
            if re.match("VRChat.exe", src_out.proplist["application.name"]):
                vrchat_source = src_out.source
                break

        if vrchat_source is None:
            return

        for src in p.source_list():
            if src.index == vrchat_source:
                return src.name
        return None
