import argparse
from collections.abc import Callable

type SubParsers = argparse._SubParsersAction[argparse.ArgumentParser]
type CommandHandler = Callable[[argparse.Namespace], int]


def add_command(
    subparsers: SubParsers, name: str, help_text: str, handler: CommandHandler
) -> argparse.ArgumentParser:
    command_parser = subparsers.add_parser(name, help=help_text)
    command_parser.set_defaults(handler=handler)
    return command_parser


def add_persona_id_argument(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument("persona_id", help="Registered spirit slug")


def print_banner(title: str, width: int = 70) -> None:
    print("=" * width)
    print(f" {title}")
    print("=" * width)
