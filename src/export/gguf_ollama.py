import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from common.errors import EvaiError


@dataclass(frozen=True)
class GGUFExportConfig:
    model_dir: Path
    output_path: Path
    quantization_type: str = "q4_k_m"


@dataclass(frozen=True)
class OllamaModelConfig:
    model_name: str
    gguf_path: Path
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str | None = None


def _require_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise EvaiError(
            f"Required external tool '{name}' was not found on PATH. "
            f"This project does not install llama.cpp/Ollama automatically; "
            f"install it and ensure '{name}' is reachable before exporting."
        )
    return resolved


def convert_to_gguf(config: GGUFExportConfig) -> Path:
    converter = _require_tool("convert_hf_to_gguf.py")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            converter,
            str(config.model_dir),
            "--outfile",
            str(config.output_path),
            "--outtype",
            "f16",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EvaiError(
            f"convert_hf_to_gguf.py failed (exit {result.returncode}): {result.stderr}"
        )
    return config.output_path


def quantize_gguf(input_path: Path, output_path: Path, quantization_type: str) -> Path:
    quantizer = _require_tool("llama-quantize")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [quantizer, str(input_path), str(output_path), quantization_type],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EvaiError(f"llama-quantize failed (exit {result.returncode}): {result.stderr}")
    return output_path


def export_gguf(config: GGUFExportConfig) -> Path:
    intermediate_path = config.output_path.with_suffix(".f16.gguf")
    convert_to_gguf(GGUFExportConfig(config.model_dir, intermediate_path, config.quantization_type))
    quantize_gguf(intermediate_path, config.output_path, config.quantization_type)
    intermediate_path.unlink(missing_ok=True)
    return config.output_path


def generate_ollama_modelfile(config: OllamaModelConfig) -> str:
    lines = [
        f"FROM {config.gguf_path}",
        f"PARAMETER temperature {config.temperature}",
        f"PARAMETER top_p {config.top_p}",
    ]
    if config.system_prompt:
        escaped = config.system_prompt.replace('"""', '\\"\\"\\"')
        lines.append(f'SYSTEM """{escaped}"""')
    return "\n".join(lines) + "\n"


def register_ollama_model(config: OllamaModelConfig, modelfile_dir: Path) -> None:
    ollama_bin = _require_tool("ollama")
    modelfile_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = modelfile_dir / f"{config.model_name}.Modelfile"
    modelfile_path.write_text(generate_ollama_modelfile(config), encoding="utf-8")

    result = subprocess.run(
        [ollama_bin, "create", config.model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EvaiError(f"ollama create failed (exit {result.returncode}): {result.stderr}")
