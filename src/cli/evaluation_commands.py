import argparse
import json

from cli.command_registry import SubParsers, add_command, add_persona_id_argument, print_banner
from common.paths import REPORTS_DIR, spirit_adapter_dir


def cmd_evaluate(args: argparse.Namespace) -> int:
    from adapter.spirit_adapter import SpiritRuntime
    from evaluation.regression import run_regression_evaluation
    from spirit_dataset.runtime_prompt import (
        load_spirit_prompt_source,
    )
    from training.spirit_lora import roster_slugs

    persona_id: str = args.persona_id
    love_level: int = args.love_level
    adapter_dir = spirit_adapter_dir(persona_id)
    if not (adapter_dir / "adapter_config.json").exists():
        print(f"! No spirit LoRA adapter found for '{persona_id}': {adapter_dir}")
        print(f"  Run `garnet-evai train-spirit --spirit {persona_id}` first.")
        return 1

    print_banner(f"[evaluate] Fixed identity-regression evaluation: spirit '{persona_id}'")

    source = load_spirit_prompt_source(persona_id)
    other_slug = next((slug for slug in roster_slugs() if slug != persona_id), None)
    other_names = [load_spirit_prompt_source(other_slug).name] if other_slug is not None else []

    runtime = SpiritRuntime([persona_id])
    profile = source.profile
    report = run_regression_evaluation(
        runtime, source, profile["fields"], other_names, love_level,
        adapter_version=str(adapter_dir), base_model_name=runtime.model.config._name_or_path,
    )
    metrics = report.metrics

    passed = sum(1 for m in metrics if m.score == 1.0)
    for case in report.cases:
        metric = case.metric
        status = (
            "NEEDS_REVIEW" if metric.score is None else "PASS" if metric.score == 1.0 else "FAIL"
        )
        print(f"  [{status}] {metric.category.value}: {metric.details}")
        print(f"    Q: {case.prompt.prompt}")
        print(f"    A: {case.response}")

    print("-" * 70)
    print(f"Result: {passed}/{len(metrics)} objective checks passed.")
    print("Semantic identity results require separate assessment.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{persona_id}_evaluation.json"
    report_path.write_text(
        json.dumps(
            {
                **report.to_dict(),
                "adapter_dir": str(adapter_dir),
                "love_level": love_level,
                "passed": passed,
                "total": len(metrics),
                "failed": sum(m.score == 0.0 for m in metrics),
                "needs_review": sum(m.score is None for m in metrics),
                "canonical_profile": profile,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report written to {report_path}")

    return 0 if passed == len(metrics) else 1


def register(subparsers: SubParsers) -> None:
    command_parser = add_command(
        subparsers,
        "evaluate",
        "Run the fixed identity-regression evaluation for one spirit LoRA adapter",
        cmd_evaluate,
    )
    add_persona_id_argument(command_parser)
    command_parser.add_argument(
        "--love-level", type=int, default=1, help="Bond level used in the memory prompt (1..40)"
    )
