"""Load the CUDA operator library compiled from ``csrc`` with the installed CUDA toolkit."""

import sys
from functools import cache
from pathlib import Path
from typing import Any

import torch

from lfm2_kernels.exceptions import Lfm2KernelsUnsupportedError

LIBRARY_NAME = "lfm2_kernels_cuda.dll" if sys.platform == "win32" else "liblfm2_kernels_cuda.so"
LIBRARY_PATH = Path(__file__).with_name(LIBRARY_NAME)
SOURCE_DIR = Path(__file__).with_name("csrc")


@cache
def operators() -> Any:
    if not LIBRARY_PATH.parent.is_dir():
        raise Lfm2KernelsUnsupportedError(
            f"The native operator library cannot load from an archive: {LIBRARY_PATH.parent}. "
            "Unpack the wheel and put the unpacked directory on the import path"
        )
    if not LIBRARY_PATH.is_file():
        raise Lfm2KernelsUnsupportedError(
            f"CUDA operator library is not built: {LIBRARY_PATH}. "
            f"Configure and build it with xmake inside {SOURCE_DIR}"
        )
    torch.ops.load_library(str(LIBRARY_PATH))
    return torch.ops.lfm2_kernels
