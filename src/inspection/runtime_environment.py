import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Any

REQUIRED_LIBRARIES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "trl",
    "safetensors",
)


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_memory_bytes: int

    @property
    def total_memory_gb(self) -> float:
        return round(self.total_memory_bytes / (1024**3), 2)


@dataclass(frozen=True)
class LibraryStatus:
    name: str
    installed: bool
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class EnvironmentReport:
    python_version: str
    python_executable: str
    platform_name: str
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_available: bool
    gpu_count: int
    gpus: list[GpuInfo]
    libraries: dict[str, LibraryStatus]
    runtime_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "platform": self.platform_name,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cuda_available": self.cuda_available,
            "gpu_count": self.gpu_count,
            "runtime_error": self.runtime_error,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_memory_bytes": g.total_memory_bytes,
                    "total_memory_gb": g.total_memory_gb,
                }
                for g in self.gpus
            ],
            "libraries": {
                name: {
                    "installed": lib.installed,
                    "version": lib.version,
                    "error": lib.error,
                }
                for name, lib in self.libraries.items()
            },
        }


def check_library(name: str) -> LibraryStatus:
    try:
        ver = metadata.version(name)
        return LibraryStatus(name=name, installed=True, version=ver)
    except metadata.PackageNotFoundError:
        return LibraryStatus(name=name, installed=False, error="Package not installed")
    except Exception as err:
        return LibraryStatus(name=name, installed=False, error=str(err))


def inspect_environment() -> EnvironmentReport:
    py_ver = sys.version.split()[0]
    py_exec = sys.executable
    plat = platform.platform()

    torch_ver: str | None = None
    torch_cuda: str | None = None
    cuda_avail = False
    gpus: list[GpuInfo] = []
    runtime_error: str | None = None

    try:
        import torch

        torch_ver = torch.__version__
        torch_cuda = getattr(torch.version, "cuda", None)
        cuda_avail = torch.cuda.is_available()

        if cuda_avail:
            device_count = torch.cuda.device_count()
            for idx in range(device_count):
                name = torch.cuda.get_device_name(idx)
                mem = torch.cuda.get_device_properties(idx).total_memory
                gpus.append(GpuInfo(index=idx, name=name, total_memory_bytes=mem))
    except Exception as err:
        cuda_avail = False
        runtime_error = f"{type(err).__name__}: {err}"

    libraries: dict[str, LibraryStatus] = {}
    for lib_name in REQUIRED_LIBRARIES:
        libraries[lib_name] = check_library(lib_name)

    return EnvironmentReport(
        python_version=py_ver,
        python_executable=py_exec,
        platform_name=plat,
        torch_version=torch_ver,
        torch_cuda_version=torch_cuda,
        cuda_available=cuda_avail,
        gpu_count=len(gpus),
        gpus=gpus,
        libraries=libraries,
        runtime_error=runtime_error,
    )


def validate_environment(
    report: EnvironmentReport | None = None,
) -> tuple[bool, list[str]]:
    env = report if report is not None else inspect_environment()
    defects: list[str] = []

    if env.runtime_error is not None:
        defects.append(f"PyTorch runtime inspection failed: {env.runtime_error}")

    if not env.cuda_available:
        defects.append("PyTorch runtime CUDA is not available. GPU acceleration is required.")

    for name in REQUIRED_LIBRARIES:
        status = env.libraries.get(name)
        if status is None or not status.installed:
            reason = status.error if status is not None else "No package status available"
            defects.append(f"Required library '{name}' is unavailable: {reason}")

    return (len(defects) == 0, defects)
