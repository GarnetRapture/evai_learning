import argparse
import json

from cli.command_registry import SubParsers, add_command, add_persona_id_argument, print_banner
from common.paths import DATA_DIR, MERGED_DIR, REPORTS_DIR
from persona.loader import discover_persona_files, load_persona_file


def cmd_evaluate(args: argparse.Namespace) -> int:
    from common.device import select_torch_device
    from evaluation.regression import build_fixed_regression_prompts, score_response_against_prompt
    from inference.generation import generate_reply
    from inference.model_loader import load_model_and_tokenizer

    persona_id: str = args.persona_id
    model_dir = MERGED_DIR / persona_id
    if not model_dir.exists():
        print(f"! No trained model found for persona '{persona_id}': {model_dir}")
        print(f"  Run `garnet-evai train {persona_id}` first.")
        return 1

    print_banner(f"[evaluate] Fixed identity-regression evaluation: persona '{persona_id}'")

    persona = load_persona_file(DATA_DIR / f"{persona_id}.json")
    other_file = next((fp for fp in discover_persona_files() if fp.stem != persona_id), None)
    other_names = [load_persona_file(other_file).name] if other_file is not None else []

    model, tokenizer = load_model_and_tokenizer(model_dir, select_torch_device())

    prompts = build_fixed_regression_prompts(persona.name, other_names)
    metrics = [
        score_response_against_prompt(generate_reply(model, tokenizer, p.prompt), p)
        for p in prompts
    ]

    passed = sum(1 for m in metrics if m.score == 1.0)
    for metric in metrics:
        status = "PASS" if metric.score == 1.0 else "FAIL"
        print(f"  [{status}] {metric.category.value}: {metric.details}")

    print("-" * 70)
    print(f"Result: {passed}/{len(metrics)} regression categories passed.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{persona_id}_evaluation.json"
    report_path.write_text(
        json.dumps(
            {
                "persona_id": persona_id,
                "persona_name": persona.name,
                "passed": passed,
                "total": len(metrics),
                "metrics": [
                    {"category": m.category.value, "score": m.score, "details": m.details}
                    for m in metrics
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report written to {report_path}")

    return 0 if passed == len(metrics) else 1


def register(subparsers: SubParsers) -> None:
    add_persona_id_argument(
        add_command(
            subparsers,
            "evaluate",
            "Run the fixed identity-regression evaluation for one trained persona model",
            cmd_evaluate,
        )
    )
