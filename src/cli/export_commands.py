import argparse

from cli.command_registry import SubParsers, add_command, add_persona_id_argument, print_banner
from common.paths import PROJECT_ROOT
from export.spirit_bundle import export_spirit


def cmd_export(args: argparse.Namespace) -> int:
    slug: str = args.persona_id
    print_banner(f"[export] Shared base and selected adapter: '{slug}'")
    path = export_spirit(slug, PROJECT_ROOT / "tmp-codex" / f"{slug}_export.json")
    print(f"Export manifest: {path}")
    return 0


def register(subparsers: SubParsers) -> None:
    add_persona_id_argument(
        add_command(
            subparsers,
            "export",
            "Export a shared-base spirit adapter manifest",
            cmd_export,
        )
    )
