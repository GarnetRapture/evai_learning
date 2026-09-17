from pathlib import Path

from common.errors import ConfigurationError
from common.paths import spirit_adapter_dir
from training.config import SpiritLoraConfig, load_spirit_lora_config
from training.spirit_lora import SpiritLoraTrainer


def load_training_config(config_path: Path | None = None) -> SpiritLoraConfig:
    return load_spirit_lora_config(config_path)


def train_persona_model(
    persona_id: str,
    config: SpiritLoraConfig | None = None,
    output_dir: Path | None = None,
) -> Path:
    target = spirit_adapter_dir(persona_id)
    if output_dir is not None and output_dir.resolve() != target.resolve():
        raise ConfigurationError("Spirit adapters use the canonical adapter directory", output_dir)
    trainer = SpiritLoraTrainer(config if config is not None else load_training_config())
    trainer.train_spirit(persona_id)
    return target
