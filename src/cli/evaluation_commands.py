import argparse
import json

from cli.command_registry import SubParsers, add_command, add_persona_id_argument, print_banner
from common.paths import MODEL_DIR, REPORTS_DIR
from evaluation.regression import run_regression_evaluation
from inference.spirit_runtime import SpiritRuntime
from spirit_dataset.roster import roster_slugs
from spirit_dataset.runtime_prompt import load_spirit_prompt_source


def cmd_evaluate(args: argparse.Namespace) -> int:
    persona_id: str = args.persona_id

    print_banner(f"[evaluate] Fixed identity-regression evaluation: spirit '{persona_id}'")

    source = load_spirit_prompt_source(persona_id)
    other_slug = next((slug for slug in roster_slugs() if slug != persona_id), None)
    other_names = [load_spirit_prompt_source(other_slug).name] if other_slug is not None else []

    runtime = SpiritRuntime([persona_id])
    profile = source.profile
    report = run_regression_evaluation(
        runtime,
        source,
        profile["fields"],
        other_names,
        weights_sha256=runtime.contract["weights_sha256"],
        base_model_name=runtime.model.config._name_or_path,
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
                "model_dir": str(MODEL_DIR),
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
        "Review the selected spirit's actual responses from the single trained model",
        cmd_evaluate,
    )
    add_persona_id_argument(command_parser)
