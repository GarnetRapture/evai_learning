import argparse
import sys

from cli.command_registry import SubParsers, add_command


def cmd_serve(args: argparse.Namespace) -> int:
    from web.server import run_server

    try:
        run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0
    except Exception as err:
        print(f"\n[error] Failed to start web chat server: {err}", file=sys.stderr)
        return 1


def register(subparsers: SubParsers) -> None:
    serve_parser = add_command(
        subparsers, "serve", "Launch the local EVAI web chat UI preview HTTP server", cmd_serve
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the server to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    serve_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the default web browser",
    )
