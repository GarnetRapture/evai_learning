import argparse
from contextlib import closing

from cli.command_registry import SubParsers, add_command, print_banner
from common.paths import GENERAL_CORPUS_FILE, INTIMACY_PATTERNS_FILE
from external_dialogue.pattern_audit import AUDIT_REPORT_FILE, audit_patterns
from external_dialogue.patterns import build_intimacy_patterns
from general_corpus.store import write_general_corpus
from spirit_dataset.builder import SpiritDatasetBuilder


def cmd_build_dataset(args: argparse.Namespace) -> int:
    print_banner("[build-dataset] Spirit-owned curriculum for the single model from docs/tbl")
    slugs = set(args.spirit) if args.spirit else None
    with closing(SpiritDatasetBuilder()) as builder:
        result = builder.build_all(slugs)

    for summary in result.summaries:
        print(
            f"  * {summary.identity.slug:<28} hero={summary.identity.hero_no:<5} "
            f"records={summary.record_count:<5} excluded={summary.exclusion_count:<5} "
            f"episodic_memory={summary.episodic_memory_count:<3} "
            f"(train={summary.train_count}/val={summary.validation_count}/"
            f"test={summary.test_count})"
        )
    print("-" * 70)
    print(f"* Spirits in roster: {len(result.roster.spirits)}")
    print(f"* Spirits built: {len(result.summaries)}")
    print(f"* Total records: {sum(item.record_count for item in result.summaries)}")
    print(f"* Total exclusions: {sum(item.exclusion_count for item in result.summaries)}")

    if slugs is not None:
        missing = slugs - {summary.identity.slug for summary in result.summaries}
        for slug in sorted(missing):
            print(f"! Unknown spirit slug: {slug}")
        if missing:
            return 1
    if result.roster.unmatched_legacy_files:
        print("\n! [legacy persona files outside the collectable roster]")
        for path in result.roster.unmatched_legacy_files:
            print(f"  - {path.name}")
    if result.spirits_without_records:
        print("\n! [spirits without canonical records]")
        for identity in result.spirits_without_records:
            print(f"  - {identity.slug} ({identity.hero_no})")
        return 1
    return 0


def cmd_build_general_corpus(args: argparse.Namespace) -> int:
    print_banner("[build-general-corpus] General dialogue and judgment patterns (ko/en/zh_tw)")
    counts = write_general_corpus()
    for key, count in sorted(counts.items()):
        print(f"  * {key:<40} {count:>10,}")
    print(f"* Conversations: {sum(counts.values()):,}")
    print(f"* File: {GENERAL_CORPUS_FILE} ({GENERAL_CORPUS_FILE.stat().st_size:,} bytes)")
    return 0


def cmd_build_dialogue_patterns(args: argparse.Namespace) -> int:
    print_banner("[build-dialogue-patterns] Shared intimacy patterns with spirit/savior slots")
    stats = build_intimacy_patterns()
    for key, count in sorted(stats.items()):
        print(f"  * {key:<40} {count:>10,}")
    print(f"* File: {INTIMACY_PATTERNS_FILE} ({INTIMACY_PATTERNS_FILE.stat().st_size:,} bytes)")
    return 0


def cmd_audit_dialogue_patterns(args: argparse.Namespace) -> int:
    print_banner("[audit-dialogue-patterns] World, gender, name and policy conflicts")
    report = audit_patterns()
    print(f"  * totals {report['totals']}")
    for check in report["checks"]:
        print(f"  * {check['name']:<22} {check['count']:>7,}  {check['description']}")
        for sample in check["samples"][: args.samples]:
            role, pattern = sample.get("role", ""), sample.get("pattern", "")
            print(f"      - [{role}] {sample['match']} | {pattern}")
            if sample.get("original"):
                print(f"        원문: {sample['original']}")
    print(f"* Report: {AUDIT_REPORT_FILE}")
    return 0


def register(subparsers: SubParsers) -> None:
    audit = add_command(
        subparsers,
        "audit-dialogue-patterns",
        "Audit shared dialogue patterns for conflicts with the spirit world and policy",
        cmd_audit_dialogue_patterns,
    )
    audit.add_argument("--samples", type=int, default=3)
    add_command(
        subparsers,
        "build-dialogue-patterns",
        "Extract shared, setting-free dialogue patterns rendered later in each spirit's speech",
        cmd_build_dialogue_patterns,
    )
    add_command(
        subparsers,
        "build-general-corpus",
        "Convert downloaded general datasets into one compact training corpus",
        cmd_build_general_corpus,
    )
    command_parser = add_command(
        subparsers,
        "build-dataset",
        "Build spirit-owned SFT records (memory + situation + canonical line) from docs/tbl",
        cmd_build_dataset,
    )
    command_parser.add_argument(
        "--spirit",
        action="append",
        help="Spirit slug to build (repeatable); builds every collectable spirit when omitted",
    )
