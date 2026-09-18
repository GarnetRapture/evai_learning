"""Capability probing and launch-tile selection for the fused short-convolution."""

from dataclasses import dataclass
from functools import cache

import torch

from shortconv_triton.exceptions import ShortConvDeviceError, ShortConvShapeError

AMPERE_MAJOR = 8


@dataclass(frozen=True)
class TileConfig:
    """One launch geometry: token rows, channel columns and warp count per program."""

    block_tokens: int
    block_channels: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class DeviceProfile:
    """The fixed properties this library tunes its launch geometry against."""

    name: str
    capability: tuple[int, int]
    multiprocessors: int

    @property
    def supports_fused_path(self) -> bool:
        return self.capability[0] >= AMPERE_MAJOR


@cache
def triton_is_importable() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


@cache
def device_profile(index: int = 0) -> DeviceProfile | None:
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(index)
    return DeviceProfile(
        name=properties.name,
        capability=(properties.major, properties.minor),
        multiprocessors=properties.multi_processor_count,
    )


def fused_path_is_available(index: int = 0) -> bool:
    profile = device_profile(index)
    return profile is not None and profile.supports_fused_path and triton_is_importable()


def require_same_cuda_device(*tensors: torch.Tensor | None) -> torch.device:
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        raise ShortConvDeviceError("The fused short-convolution requires at least one operand")
    device = present[0].device
    if device.type != "cuda":
        raise ShortConvDeviceError(f"The fused short-convolution requires CUDA operands: {device}")
    for tensor in present[1:]:
        if tensor.device != device:
            raise ShortConvDeviceError(
                f"Operands span several devices: {device} and {tensor.device}"
            )
    return device


def require_contiguous_last_dim(tensor: torch.Tensor, name: str) -> None:
    if tensor.stride(-1) != 1:
        raise ShortConvShapeError(f"{name} must be contiguous along its channel axis")


def select_tile(
    batch: int, seq_len: int, channels: int, index: int = 0
) -> TileConfig:
    """Pick token/channel tiles that keep every multiprocessor occupied.

    The convolution is memory bound, so the tile is chosen to cover the device with
    resident programs rather than to maximise arithmetic reuse inside one program.
    """
    profile = device_profile(index)
    multiprocessors = profile.multiprocessors if profile is not None else 1
    block_channels = 64 if channels % 64 == 0 else 32
    for block_tokens in (128, 64, 32, 16):
        token_blocks = -(-seq_len // block_tokens)
        channel_blocks = -(-channels // block_channels)
        if batch * token_blocks * channel_blocks >= multiprocessors * 2:
            return TileConfig(
                block_tokens=block_tokens,
                block_channels=block_channels,
                num_warps=4 if block_tokens * block_channels >= 4096 else 2,
                num_stages=2,
            )
    return TileConfig(block_tokens=16, block_channels=block_channels, num_warps=2, num_stages=2)
