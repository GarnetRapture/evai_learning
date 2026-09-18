"""Errors raised by this library alone, so it carries no host-project dependency."""


class ShortConvError(RuntimeError):
    """Base class for every failure raised by shortconv_triton."""


class ShortConvUnsupportedError(ShortConvError):
    """The running environment cannot execute the fused Triton path."""


class ShortConvShapeError(ShortConvError):
    """Operand ranks, sizes or strides do not match the fused contract."""


class ShortConvDeviceError(ShortConvError):
    """Operands are not all resident on one CUDA device."""
