"""Errors raised by this library alone, so it carries no host-project dependency."""


class Lfm2KernelsError(RuntimeError):
    """Base class for every failure raised by lfm2_kernels."""


class Lfm2KernelsUnsupportedError(Lfm2KernelsError):
    """The running environment or request cannot execute on the CUDA operator library."""


class Lfm2KernelsShapeError(Lfm2KernelsError):
    """Operand ranks, sizes or dtypes do not match the operator contract."""


class Lfm2KernelsDeviceError(Lfm2KernelsError):
    """Operands are not all resident on one CUDA device."""


class Lfm2KernelsNonFiniteError(Lfm2KernelsError):
    """A gradient norm became non-finite; the optimizer withheld that update."""
