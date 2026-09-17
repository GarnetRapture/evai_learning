import argparse
from contextlib import closing

from cli.command_registry import SubParsers, add_command, print_banner
from spirit_dataset.builder import SpiritDatasetBuilder


def cmd_build_dataset(args: argparse.Namespace) -> int:
    print_banner("[build-dataset] Spirit-owned curriculum for the single model from data/tbl")
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


def register(subparsers: SubParsers) -> None:
    command_parser = add_command(
        subparsers,
        "build-dataset",
        "Build spirit-owned SFT records (memory + situation + canonical line) from data/tbl",
        cmd_build_dataset,
    )
    command_parser.add_argument(
        "--spirit",
        action="append",
        help="Spirit slug to build (repeatable); builds every collectable spirit when omitted",
    )
