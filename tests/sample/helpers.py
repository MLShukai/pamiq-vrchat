import logging

import pytest
import torch

logger = logging.getLogger(__name__)

CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda:0")


def get_available_devices() -> list[torch.device]:
    devices = [CPU_DEVICE]
    if torch.cuda.is_available():
        devices.append(CUDA_DEVICE)
    return devices


logger.info("Available devices: " + ", ".join(map(str, get_available_devices())))

parametrize_device = pytest.mark.parametrize("device", get_available_devices())
