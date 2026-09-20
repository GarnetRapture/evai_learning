"""Load the CUDA operator library compiled from ``csrc`` with the installed CUDA toolkit."""

import sys
from functools import cache
from pathlib import Path
from typing import Any

import torch

from evai_kernels.exceptions import EvaiKernelsUnsupportedError

LIBRARY_NAME = "evai_kernels_cuda.dll" if sys.platform == "win32" else "libevai_kernels_cuda.so"
LIBRARY_PATH = Path(__file__).with_name(LIBRARY_NAME)
SOURCE_DIR = Path(__file__).with_name("csrc")


@cache
def operators() -> Any:
    if not LIBRARY_PATH.parent.is_dir():
        raise EvaiKernelsUnsupportedError(
            f"The native operator library cannot load from an archive: {LIBRARY_PATH.parent}. "
            "Unpack the wheel and put the unpacked directory on the import path"
        )
    if not LIBRARY_PATH.is_file():
        raise EvaiKernelsUnsupportedError(
            f"CUDA operator library is not built: {LIBRARY_PATH}. "
            f"Configure and build it with xmake inside {SOURCE_DIR}"
        )
    torch.ops.load_library(str(LIBRARY_PATH))
    return torch.ops.evai_kernels
