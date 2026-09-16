"""GarnetRapture_evai: EVAI persona fine-tuning pipeline for LiquidAI/LFM2.5-230M-Base.

Top-level package exports and CLI entrypoint.
"""

from . import (
    cli,
    dataset,
    dialogue,
    environment,
    errors,
    evaluate,
    export,
    loader,
    manifest,
    model,
    normalize,
    paths,
    schema,
    split,
    train,
)
from .cli import main

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "main",
    "cli",
    "dataset",
    "dialogue",
    "environment",
    "errors",
    "evaluate",
    "export",
    "loader",
    "manifest",
    "model",
    "normalize",
    "paths",
    "schema",
    "split",
    "train",
]
