"""PC deployment references the one trained model; no weight copies."""

import json
from pathlib import Path

from common.errors import EvaiError
from common.hashing import compute_file_sha256
from common.model_contract import MODEL_ID, read_training_contract, verify_backbone
from common.paths import ARTIFACT_DIR, MODEL_DIR
from inference.spirit_runtime import SpiritRuntime
from spirit_dataset.runtime_prompt import spirit_file_path

RUNTIME_MANIFEST = ARTIFACT_DIR / "runtime.json"


def build_pc() -> Path:
    contract = read_training_contract(MODEL_DIR)
    if contract is None:
        raise EvaiError("Train the single model before building its PC manifest")
    digest = verify_backbone(MODEL_DIR)
    for slug in contract["spirits"]:
        if compute_file_sha256(spirit_file_path(slug)) != contract["profile_sha256"][slug]:
            raise EvaiError(f"Spirit profile changed since model training: {slug}")
    RUNTIME_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_MANIFEST.write_text(
        json.dumps(
            {
                "base_model": MODEL_ID,
                "weights_sha256": digest,
                "spirits": contract["spirits"],
                "quality_approved": contract["quality_approved"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return RUNTIME_MANIFEST


def load_pc_runtime(path: Path, *, use_gguf: bool = False) -> SpiritRuntime:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    contract = read_training_contract(MODEL_DIR)
    if (
        contract is None
        or manifest["base_model"] != MODEL_ID
        or manifest["weights_sha256"] != contract["weights_sha256"]
    ):
        raise EvaiError("PC manifest does not reference the current trained model")
    return SpiritRuntime(manifest["spirits"], use_gguf=use_gguf)
