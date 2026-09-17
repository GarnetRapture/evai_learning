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

    print_banner(f"[train] Independent LoRA SFT: spirit '{persona_id}'")

    output_dir = train_persona_model(persona_id)

    print("-" * 70)
    print(f"Result: training complete. Adapter saved to {output_dir}")
    return 0


def cmd_train_spirit(args: argparse.Namespace) -> int:
    from common.errors import EvaiError
    from common.paths import spirit_training_report_path
    from training.spirit_lora import SpiritLoraTrainer, load_spirit_lora_config, roster_slugs

    known = roster_slugs()
    slugs: list[str] = list(dict.fromkeys(args.spirit)) if args.spirit else known
    unknown = [slug for slug in slugs if slug not in known]
    if unknown:
        print(f"! Unknown spirit slug(s): {', '.join(unknown)}")
        return 1

    config = load_spirit_lora_config(rank_override=args.rank)
    print_banner(
        f"[train-spirit] LoRA ranks={config.lora.rank_candidates} "
        f"on LFM2.5-230M: {len(slugs)} spirit(s)"
    )
    trainer = SpiritLoraTrainer(config)
    failures: list[str] = []
    for slug in slugs:
        try:
            report = trainer.train_spirit(slug)
        except EvaiError as err:
            print(f"  ! {slug}: {err}")
            failures.append(slug)
            continue
        last = report.epochs[-1]
        base = (
            "n/a" if report.base_validation_loss is None else f"{report.base_validation_loss:.3f}"
        )
        validation = "n/a" if last.validation_loss is None else f"{last.validation_loss:.3f}"
        print(
            f"  * {slug:<28} train={report.train_examples:<4} val_loss {base} -> {validation} "
            f"train_loss={last.train_loss:.3f} {report.seconds:.1f}s"
        )
        print(f"    report: {spirit_training_report_path(slug)}")
    print("-" * 70)
    print(f"* Trained: {len(slugs) - len(failures)} / {len(slugs)}")
    for slug in failures:
        print(f"! Failed: {slug}")
    return 1 if failures else 0


def register(subparsers: SubParsers) -> None:
    spirit_parser = add_command(
        subparsers,
        "train-spirit",
        "Train per-spirit LoRA adapters on the shared LFM2.5-230M base (base loaded once)",
        cmd_train_spirit,
    )
    spirit_parser.add_argument(
        "--spirit",
        action="append",
        help="Spirit slug to train (repeatable); trains every roster spirit when omitted",
    )
    spirit_parser.add_argument(
        "--rank", type=int, default=None, help="Override LoRA rank (alpha = 2 x rank)"
    )

    add_persona_id_argument(
        add_command(
            subparsers,
            "train",
            "Train one spirit LoRA adapter on the shared LFM2.5-230M base",
            cmd_train,
        )
    )
