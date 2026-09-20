"""Device contract for the operator library compiled for ``sm_86`` with ``compute_86`` PTX."""

from dataclasses import dataclass
from functools import cache

import torch

from evai_kernels.exceptions import EvaiKernelsDeviceError, EvaiKernelsUnsupportedError

MINIMUM_CAPABILITY = (8, 6)


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    capability: tuple[int, int]
    multiprocessors: int


@cache
def device_profile(index: int) -> DeviceProfile:
    properties = torch.cuda.get_device_properties(index)
    return DeviceProfile(
        name=properties.name,
        capability=(properties.major, properties.minor),
        multiprocessors=properties.multi_processor_count,
    )


def require_same_cuda_device(*tensors: torch.Tensor | None) -> torch.device:
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        raise EvaiKernelsDeviceError("The CUDA operators require at least one operand")
    device = present[0].device
    if device.type != "cuda":
        raise EvaiKernelsDeviceError(f"The CUDA operators require CUDA operands: {device}")
    for tensor in present[1:]:
        if tensor.device != device:
            raise EvaiKernelsDeviceError(
                f"Operands span several devices: {device} and {tensor.device}"
            )
    profile = device_profile(device.index or 0)
    if profile.capability < MINIMUM_CAPABILITY:
        raise EvaiKernelsUnsupportedError(
            f"{profile.name} has compute capability {profile.capability}; "
            f"the library is built for {MINIMUM_CAPABILITY} and newer"
        )
    return device
