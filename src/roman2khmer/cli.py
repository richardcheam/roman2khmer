"""`roman2khmer`: convert from the command line, download the model, or run a small HTTP API.

    roman2khmer convert "bong mok nham bay nov"            # one message -> Khmer on stdout
    printf 'knhom tov sala\\nok bong\\n' | roman2khmer convert  # one message per input line, streamed
    roman2khmer download                                   # fetch the pinned model files into the cache
    roman2khmer serve --port 8000                          # POST /convert {"text": ...} (needs [serve])

The model comes from `--model DIR` (a folder with the model files) or, by default, the pinned release on Hugging Face
(`pip install "roman2khmer[hub]"`; add `--offline` to use only the local cache). Results go to stdout (UTF-8), messages
to stderr. Exit status: 0 on success, 1 if a conversion failed, 2 for usage, model or download errors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load(args):
    from roman2khmer import Converter
    settings = {"gamma": 0.0} if args.no_recheck else {}
    if args.model:
        return Converter.from_dir(Path(args.model), **settings)
    return Converter.from_pretrained(local_files_only=args.offline, **settings)


def _convert(args) -> int:
    conv = _load(args)
    out = sys.stdout
    if args.text:
        print(conv.convert(" ".join(args.text)), file=out)
        return 0
    status = 0
    for line in sys.stdin:
        text = line.rstrip("\n")
        try:
            print(conv.convert(text), file=out, flush=True)
        except Exception as e:          # keep going: one bad line must not stop a stream
            print(f"error on input line: {e}", file=sys.stderr)
            print("", file=out, flush=True)
            status = 1
    return status


def _download(args) -> int:
    from roman2khmer import hub
    print(hub.download(cache_dir=args.cache_dir))
    return 0


def _serve(args) -> int:
    try:
        import uvicorn
        from roman2khmer.server import create_app
    except ImportError:
        print('serving needs the [serve] extra: pip install "roman2khmer[serve]"', file=sys.stderr)
        return 2
    conv = _load(args)
    app = create_app(conv, max_chars=args.max_chars, max_waiting=args.max_waiting, timeout_s=args.timeout)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"listening on {args.host}: put authentication, TLS and rate limits in front of it before exposing "
              "it to other people", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, timeout_keep_alive=5)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="roman2khmer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_opts(p):
        p.add_argument("--model", help="folder with translit.npz, reverse.npz, khmer_bigram.npz "
                                       "(default: the pinned release from Hugging Face)")
        p.add_argument("--offline", action="store_true", help="use only already-downloaded model files")
        p.add_argument("--no-recheck", action="store_true", help="skip the recheck (about twice as fast)")

    c = sub.add_parser("convert", help="convert text (arguments, or stdin line by line)")
    c.add_argument("text", nargs="*", help="the message; omit to read stdin")
    model_opts(c)
    c.set_defaults(func=_convert)

    d = sub.add_parser("download", help="download the pinned model files and print their folder")
    d.add_argument("--cache-dir", help="Hugging Face cache folder")
    d.set_defaults(func=_download)

    s = sub.add_parser("serve", help="HTTP JSON API (needs [serve])")
    model_opts(s)
    s.add_argument("--host", default="127.0.0.1", help="default: this machine only")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-chars", type=int, default=1000, help="longest accepted message")
    s.add_argument("--max-waiting", type=int, default=8, help="requests allowed to wait; more get HTTP 503")
    s.add_argument("--timeout", type=float, default=30.0, help="seconds before a request gets HTTP 504")
    s.set_defaults(func=_serve)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ImportError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        if type(e).__name__ == "IntegrityError":
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
