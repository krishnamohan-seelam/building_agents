"""`record` command: a thin argparse wrapper over recorder/record.py's record_run (M2-T9)."""

import argparse

from recorder.record import record_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="record")
    parser.add_argument("example")
    parser.add_argument("--input", type=int, required=True)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--publish", type=str, default=None)
    parser.add_argument("--preview", type=str, default=None, metavar="BLOG_PATH")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.publish is not None:
        raise NotImplementedError("--publish is not implemented yet; use --preview and publish by hand.")

    trace = record_run(
        args.example, input_index=args.input, cached=args.cached, preview_path=args.preview
    )
    print(f"Recorded {trace['example']} input {args.input} -> {len(trace['steps'])} steps")
    if args.preview is not None:
        print(
            f"Copied into {args.preview}/assets/traces/ for local review "
            f"(run `bundle exec jekyll serve` there; nothing was committed)."
        )


if __name__ == "__main__":
    main()
