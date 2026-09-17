import argparse

from cli import (
    adapter_commands,
    dataset_commands,
    evaluation_commands,
    export_commands,
    inspection_commands,
    training_commands,
)

COMMAND_MODULES = (
    inspection_commands,
    dataset_commands,
    training_commands,
    evaluation_commands,
    export_commands,
    adapter_commands,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garnet-evai",
        description="EVAI spirit training pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    for module in COMMAND_MODULES:
        module.register(subparsers)
    return parser
