"""`record` command: a thin argparse wrapper over recorder/record.py's record_run (M2-T9)."""

import argparse

from recorder.record import record_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="record")
    parser.add_argument("example")
    parser.add_argument("--input", type=int, required=True)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--publish", type=str, default=None)
    parser.add_argument("--preview", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.publish is not None or args.preview:
        raise NotImplementedError(
            "--publish and --preview are M4 scope; not implemented in M2."
        )

    trace = record_run(args.example, input_index=args.input, cached=args.cached)
    print(f"Recorded {trace['example']} input {args.input} -> {len(trace['steps'])} steps")


if __name__ == "__main__":
    main()
