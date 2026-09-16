import argparse
import json

from cli.command_registry import SubParsers, add_command, add_persona_id_argument, print_banner
from common.paths import REPORTS_DIR


def cmd_adapter_fidelity(args: argparse.Namespace) -> int:
    from adapter.spirit_adapter import measure_rank_fidelity

    persona_id: str = args.persona_id
    ranks: list[int] = args.ranks
    print_banner(f"[adapter-fidelity] Full-FT vs base+LoRA logits: persona '{persona_id}'")

    report = measure_rank_fidelity(persona_id, ranks, max_records=args.max_records)
    print(f"* evaluated tokens: {report.token_count}")
    print(
        f"* base vs full-FT: mean KL={report.base_mean_kl_to_full:.5f}, "
        f"top-1 agreement={report.base_top1_agreement_with_full:.4f}"
    )
    for row in report.ranks:
        print(
            f"  rank {row.rank:>4}: mean KL={row.mean_kl_to_full:.5f}, "
            f"top-1 agreement={row.top1_agreement_with_full:.4f}, "
            f"retained effect={row.retained_effect:.4f}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{persona_id}_adapter_fidelity.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"Report written to {out_path}")
    return 0


def cmd_extract_adapter(args: argparse.Namespace) -> int:
    from adapter.spirit_adapter import extract_spirit_adapter

    persona_id: str = args.persona_id
    print_banner(f"[extract-adapter] persona '{persona_id}', rank {args.rank}")
    out_dir = extract_spirit_adapter(persona_id, args.rank)
    total = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    print(f"Result: adapter written to {out_dir} ({total / 1024 / 1024:.1f} MB)")
    return 0


def cmd_adapter_runtime_check(args: argparse.Namespace) -> int:
    from adapter.spirit_adapter import verify_runtime_fidelity

    persona_ids: list[str] = args.persona_ids
    print_banner(f"[adapter-runtime-check] single backbone + {len(persona_ids)} spirit adapters")
    for row in verify_runtime_fidelity(persona_ids, max_records=args.max_records):
        print(
            f"  {row.persona_id:<24} tokens={row.token_count:<5} "
            f"mean KL to full-FT={row.mean_kl_to_full:.5f} "
            f"top-1 agreement={row.top1_agreement_with_full:.4f}"
        )
    return 0


def register(subparsers: SubParsers) -> None:
    fidelity_parser = add_command(
        subparsers,
        "adapter-fidelity",
        "Measure full-FT vs base+LoRA logit fidelity across candidate ranks",
        cmd_adapter_fidelity,
    )
    add_persona_id_argument(fidelity_parser)
    fidelity_parser.add_argument(
        "--ranks", type=int, nargs="+", default=[16, 32, 64, 128, 256], help="Candidate ranks"
    )
    fidelity_parser.add_argument(
        "--max-records", type=int, default=16, help="Held-out SFT records to score"
    )

    extract_parser = add_command(
        subparsers,
        "extract-adapter",
        "Extract a PEFT spirit adapter from a full fine-tuned persona checkpoint",
        cmd_extract_adapter,
    )
    add_persona_id_argument(extract_parser)
    extract_parser.add_argument("--rank", type=int, required=True, help="LoRA rank")

    runtime_check_parser = add_command(
        subparsers,
        "adapter-runtime-check",
        "Load spirit adapters into one backbone and compare each against its full-FT",
        cmd_adapter_runtime_check,
    )
    runtime_check_parser.add_argument("persona_ids", nargs="+", help="Persona ids to load")
    runtime_check_parser.add_argument(
        "--max-records", type=int, default=16, help="Held-out SFT records to score"
    )
