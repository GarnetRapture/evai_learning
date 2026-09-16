import argparse
import json

from cli.command_registry import SubParsers, add_command, print_banner
from common.errors import EvaiError
from common.paths import ensure_artifact_directories
from persona.loader import discover_persona_files, load_persona_file
from sft_dataset.dialogue import extract_dialogue_exchanges
from sft_dataset.manifest import build_dataset_manifest
from sft_dataset.records import (
    build_greeting_record,
    build_persona_dataset,
    build_speech_pattern_records,
)
from sft_dataset.split import SplitConfig, leakage_safe_split
from sft_dataset.storage import persona_dataset_dir, sft_split_path, write_records_jsonl

DATASET_VERSION = "0.1.0"
DATASET_LANGUAGE = "ko"


def cmd_build_dataset(args: argparse.Namespace) -> int:
    print_banner("[build-dataset] Deterministic SFT dataset generation for every persona")

    ensure_artifact_directories()
    all_files = discover_persona_files()
    if not all_files:
        print("! No persona JSON files found under data/.")
        return 1

    split_config = SplitConfig()
    total_accepted = 0
    total_excluded = 0
    load_failures: list[str] = []
    no_data_personas: list[str] = []

    for file_path in all_files:
        persona_id = file_path.stem
        try:
            persona = load_persona_file(file_path)
        except (EvaiError, OSError) as err:
            load_failures.append(f"{file_path.name}: load failed ({err})")
            continue

        source_file = str(file_path)
        extraction = extract_dialogue_exchanges(persona, persona_id, source_file)

        dialogue_records, dialogue_exclusions = build_persona_dataset(
            extraction, persona_id, DATASET_LANGUAGE, source_file
        )
        speech_records, speech_exclusions = build_speech_pattern_records(
            persona.speech_patterns, persona_id, persona.name, DATASET_LANGUAGE, source_file
        )
        greeting_record, greeting_exclusion = build_greeting_record(
            persona.personality.greeting, persona_id, persona.name, DATASET_LANGUAGE, source_file
        )

        records = [*dialogue_records, *speech_records]
        if greeting_record is not None:
            records.append(greeting_record)

        exclusions = [*dialogue_exclusions, *speech_exclusions]
        if greeting_exclusion is not None:
            exclusions.append(greeting_exclusion)

        if not records:
            no_data_personas.append(
                f"{file_path.name}: evertalk/speech_patterns/greeting are all empty; "
                "no source material to learn from."
            )
            continue

        split = leakage_safe_split(records, split_config)
        manifest = build_dataset_manifest(
            persona_id, persona.name, file_path, DATASET_VERSION, split
        )

        write_records_jsonl(sft_split_path(persona_id, "train"), split.train)
        write_records_jsonl(sft_split_path(persona_id, "validation"), split.validation)
        write_records_jsonl(sft_split_path(persona_id, "test"), split.test)
        write_records_jsonl(sft_split_path(persona_id, "exclusions"), exclusions)
        (persona_dataset_dir(persona_id) / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        total_accepted += len(records)
        total_excluded += len(exclusions)
        print(
            f"  * {persona_id:<24} accepted={len(records):<5} excluded={len(exclusions):<5} "
            f"(train={len(split.train)}/val={len(split.validation)}/test={len(split.test)})"
        )

    print("-" * 70)
    print(f"* Persona files processed: {len(all_files)}")
    print(f"* Total accepted SFT records: {total_accepted}")
    print(f"* Total excluded (traceable) records: {total_excluded}")

    if load_failures:
        print("\n! [load failures]")
        for failure in load_failures:
            print(f"  - {failure}")

    if no_data_personas:
        print("\n! [no source material — a data limitation, nothing invented]")
        for entry in no_data_personas:
            print(f"  - {entry}")

    if load_failures or no_data_personas:
        return 1

    print("\nResult: deterministic SFT dataset generation completed for every persona.")
    return 0


def register(subparsers: SubParsers) -> None:
    add_command(
        subparsers,
        "build-dataset",
        "Build deterministic, classified, leakage-safe SFT datasets for every persona",
        cmd_build_dataset,
    )
