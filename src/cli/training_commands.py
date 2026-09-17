import argparse

from cli.command_registry import SubParsers, add_command
from training.config import load_training_config
from training.trainer import train_model


def cmd_train(args: argparse.Namespace) -> int:
    output = train_model(load_training_config())
    print(f"Single trained model: {output}", flush=True)
    print("Actual multilingual persona response review is still required.", flush=True)
    return 0


def register(subparsers: SubParsers) -> None:
    add_command(subparsers, "train", "Train all spirits jointly into the one 230M model", cmd_train)
    add_command(
        subparsers,
        "train-spirit",
        "Train the joint spirit curriculum into one model (no per-spirit weights)",
        cmd_train,
    )
