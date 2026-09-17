import json
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.paths import MODEL_DIR, PROJECT_ROOT, spirit_adapter_dir
from sft_dataset.manifest import compute_file_sha256
from spirit_dataset.runtime_prompt import spirit_file_path


def export_spirit(slug: str, output: Path) -> Path:
    adapter = spirit_adapter_dir(slug)
    files = {
        "base_weights": MODEL_DIR / "model.safetensors",
        "base_config": MODEL_DIR / "config.json",
        "adapter_weights": adapter / "adapter_model.safetensors",
        "adapter_config": adapter / "adapter_config.json",
        "spirit_profile": spirit_file_path(slug),
    }
    for path in files.values():
        if not path.is_file():
            raise EvaiError(f"Export source is missing: {path}")
    manifest: dict[str, Any] = {
        "format": "evai.base-selected-adapter.v1",
        "base_model": "LiquidAI/LFM2.5-230M-Base",
        "spirit": slug,
        "project_root": str(PROJECT_ROOT),
        "files": {key: {"path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "sha256": compute_file_sha256(path)} for key, path in files.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_exported_spirit(manifest_path: Path) -> Any:
    from adapter.spirit_adapter import SpiritRuntime

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format"] != "evai.base-selected-adapter.v1":
        raise EvaiError(f"Unsupported spirit export format: {manifest['format']}")
    root = Path(manifest["project_root"])
    for item in manifest["files"].values():
        path = root / item["path"]
        if compute_file_sha256(path) != item["sha256"]:
            raise EvaiError(f"Export source changed: {path}")
    if (root / manifest["files"]["base_config"]["path"]).parent.resolve() != MODEL_DIR.resolve():
        raise EvaiError("Export backbone must match the configured local backbone")
    adapter_root = (root / manifest["files"]["adapter_config"]["path"]).parent.parent
    return SpiritRuntime([manifest["spirit"]], adapter_root)
