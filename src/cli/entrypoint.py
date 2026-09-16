import io
import sys

from cli.parser import build_parser
from common.errors import EvaiError


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        sys.exit(0)

    try:
        sys.exit(handler(args))
    except EvaiError as err:
        print(f"\n[error] {err}", file=sys.stderr)
        sys.exit(1)
