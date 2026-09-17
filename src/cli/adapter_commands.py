"""Inspect fixed-architecture adapters; old Full-FT extraction is not a training route."""

import argparse
import json

from cli.command_registry import SubParsers, add_command, add_persona_id_argument
from common.paths import spirit_adapter_dir


def cmd_adapter_check(args: argparse.Namespace) -> int:
    from adapter.spirit_adapter import read_adapter_contract

    contract = read_adapter_contract(spirit_adapter_dir(args.persona_id), args.persona_id)
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0


def register(subparsers: SubParsers) -> None:
    add_persona_id_argument(
        add_command(
            subparsers,
            "adapter-check",
            "Inspect selected adapter/base curriculum provenance",
            cmd_adapter_check,
        )
    )
