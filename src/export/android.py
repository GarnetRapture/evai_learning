"""Build the Android native consumer with an explicitly supplied llama.cpp source tree."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from common.errors import EvaiError
from common.paths import ARTIFACT_DIR, PROJECT_ROOT, SCRATCH_DIR


def build_android_runtime(llama_cpp: Path, ndk: Path, jobs: int = 4) -> Path:
    source = llama_cpp.resolve()
    toolchain = ndk.resolve() / "build" / "cmake" / "android.toolchain.cmake"
    if not (source / "src" / "models" / "lfm2.cpp").is_file():
        raise EvaiError(f"LFM2-capable llama.cpp source is required: {source}")
    if not toolchain.is_file():
        raise EvaiError(f"Android NDK toolchain is missing: {toolchain}")
    if jobs < 1:
        raise EvaiError("Android build jobs must be positive")
    build = SCRATCH_DIR / "android-build"
    output = ARTIFACT_DIR / "android" / "arm64-v8a"
    temporary = SCRATCH_DIR / "native-tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "TEMP": str(temporary), "TMP": str(temporary)}
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DANDROID_ABI=arm64-v8a",
        "-DANDROID_PLATFORM=android-28",
        "-DGGML_NATIVE=OFF",
        "-DGGML_OPENMP=OFF",
        "-DGGML_LLAMAFILE=OFF",
        "-DLLAMA_OPENSSL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
    ]
    commands = [
        configure,
        ["cmake", "--build", str(build), "--target", "llama-cli", "-j", str(jobs)],
    ]
    try:
        for command in commands:
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
    except subprocess.CalledProcessError as error:
        raise EvaiError(f"Android native build failed with exit code {error.returncode}") from error
    binary = build / "bin" / "llama-cli"
    with binary.open("rb") as handle:
        header = handle.read(20)
    if header[:4] != b"\x7fELF" or int.from_bytes(header[18:20], "little") != 183:
        raise EvaiError(f"Native output is not an Android AArch64 ELF: {binary}")
    output.mkdir(parents=True, exist_ok=True)
    for artifact in (binary, *sorted((build / "bin").glob("*.so"))):
        shutil.copy2(artifact, output / artifact.name)
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "abi": "arm64-v8a",
                "minimum_api": 28,
                "llama_cpp_revision": revision,
                "ndk": str(ndk.resolve()),
                "backbone": "shared GGUF",
                "active_spirit_adapters": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
