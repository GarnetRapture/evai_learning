import argparse

from cli.command_registry import SubParsers, add_command, add_persona_id_argument, print_banner
from sft_dataset.storage import sft_split_path


def cmd_train(args: argparse.Namespace) -> int:
    from training.trainer import train_persona_model

    persona_id: str = args.persona_id
    dataset_path = sft_split_path(persona_id, "train")
    if not dataset_path.exists():
        print(f"! No SFT dataset found for persona '{persona_id}': {dataset_path}")
        print("  Run `garnet-evai build-dataset` first.")
        return 1

    print_banner(f"[train] Full-parameter fine-tuning: persona '{persona_id}'")

    output_dir = train_persona_model(persona_id)

    print("-" * 70)
    print(f"Result: training complete. Model saved to {output_dir}")
    return 0


def register(subparsers: SubParsers) -> None:
    add_persona_id_argument(
        add_command(
            subparsers,
            "train",
            "Run full-parameter SFT for one persona's spirit identity",
            cmd_train,
        )
    )
