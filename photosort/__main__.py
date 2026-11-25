"""CLI: `python -m photosort serve` or `python -m photosort import <folder>`."""
import argparse
import json
import logging
import os
from pathlib import Path

DEFAULT_LIBRARY = os.environ.get("PHOTOSORT_LIBRARY", str(Path(__file__).resolve().parent.parent / "library"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="photosort")
    ap.add_argument("--library", default=DEFAULT_LIBRARY, help="library directory (store + catalog)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)

    i = sub.add_parser("import", help="import a folder (blocking)")
    i.add_argument("folder")
    i.add_argument("--mode", choices=["copy", "hardlink"], default="copy")

    e = sub.add_parser("export", help="materialise a factor as a hardlink folder tree")
    e.add_argument("factor", help="date | place | camera | type")
    e.add_argument("dest")

    sub.add_parser("analyze", help="re-run faces, events, custom rules and questions")
    m = sub.add_parser("models", help="download local models: faces (~39 MB), CLIP (~600 MB), places (~1 MB)")
    m.add_argument("which", nargs="*", choices=["faces", "clip", "places"], help="default: all")

    args = ap.parse_args()
    if args.cmd == "models":
        from .clip import download_clip
        from .faces import MODELS_DIR, download_models
        from .places import download_places

        which = args.which or ["faces", "clip", "places"]
        for name, fn in (("faces", download_models), ("clip", download_clip), ("places", download_places)):
            if name in which:
                print(f"downloading {name}…", flush=True)
                fn()
        print(f"Models ready in {MODELS_DIR}. Run `python -m photosort analyze` to apply them to your library.")
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from .library import Library

    lib = Library(Path(args.library))
    if args.cmd == "serve":
        import uvicorn

        from .api import create_app

        print(f"PhotoSort library: {lib.root}\nOpen http://{args.host}:{args.port}")
        uvicorn.run(create_app(lib), host=args.host, port=args.port, log_level="warning")
    elif args.cmd == "import":
        print(json.dumps(lib.import_sync(args.folder, args.mode), indent=2))
    elif args.cmd == "analyze":
        print(json.dumps(lib.analyse_sync(), indent=2))
    elif args.cmd == "export":
        print(json.dumps(lib.export_view(args.factor, args.dest), indent=2))


if __name__ == "__main__":
    main()
