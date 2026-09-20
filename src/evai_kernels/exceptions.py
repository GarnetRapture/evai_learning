"""Errors raised by this library alone, so it carries no host-project dependency."""


class EvaiKernelsError(RuntimeError):
    """Base class for every failure raised by evai_kernels."""


class EvaiKernelsUnsupportedError(EvaiKernelsError):
    """The running environment or request cannot execute on the CUDA operator library."""


class EvaiKernelsShapeError(EvaiKernelsError):
    """Operand ranks, sizes or dtypes do not match the operator contract."""


class EvaiKernelsDeviceError(EvaiKernelsError):
    """Operands are not all resident on one CUDA device."""


class EvaiKernelsNonFiniteError(EvaiKernelsError):
    """A gradient norm became non-finite; the optimizer withheld that update."""
