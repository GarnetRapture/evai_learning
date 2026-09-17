"""Blocking byte-range ownership for the project's fixed Windows runtime."""

import ctypes
import msvcrt
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
_LOCK_FILE = _KERNEL32.LockFileEx
_LOCK_FILE.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
    wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_Overlapped),
]
_LOCK_FILE.restype = wintypes.BOOL
_UNLOCK_FILE = _KERNEL32.UnlockFileEx
_UNLOCK_FILE.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
    wintypes.DWORD, ctypes.POINTER(_Overlapped),
]
_UNLOCK_FILE.restype = wintypes.BOOL
_EXCLUSIVE_LOCK = 0x00000002


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Wait for byte zero; the OS releases ownership if the process exits."""
    with path.open("a+b") as stream:
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        offset = _Overlapped()
        if not _LOCK_FILE(handle, _EXCLUSIVE_LOCK, 0, 1, 0, ctypes.byref(offset)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not _UNLOCK_FILE(handle, 0, 1, 0, ctypes.byref(offset)):
                raise ctypes.WinError(ctypes.get_last_error())
