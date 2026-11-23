"""Command-line entrypoint for Authorship Verification (demo server, single-pair predict, model download)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlu_authorship.artifacts import DRIVE_FOLDER_URL, download_from_drive, find_artifact
from nlu_authorship.web_demo import ModelRegistry, serve_demo


def _read_text_arg(inline: str | None, file_path: str | None) -> str:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    if inline:
        return inline
    raise ValueError("Provide either --text-1/--text-2 or --text-file-1/--text-file-2.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authorship Verification (Stylometric ML & DeBERTa-v3-large)")
    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo", help="Launch interactive local web UI (ML / DL selector)")
    p_demo.add_argument("--host", default="127.0.0.1")
    p_demo.add_argument("--port", type=int, default=8000)
    p_demo.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p_demo.add_argument("--preload", default=None, choices=["ml", "dl", "both"])
    p_demo.add_argument("--no-browser", action="store_true")

    p_pred = sub.add_parser("predict", help="Score a pair of English passages from CLI")
    p_pred.add_argument("--model", default="both", choices=["ml", "dl", "both"])
    p_pred.add_argument("--text-1", default=None)
    p_pred.add_argument("--text-2", default=None)
    p_pred.add_argument("--text-file-1", default=None)
    p_pred.add_argument("--text-file-2", default=None)
    p_pred.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p_pred.add_argument("--check-symmetry", action="store_true")

    p_dl = sub.add_parser("download", help="Download public model weights from Google Drive")
    p_dl.add_argument("--model", default="ml", choices=["ml", "dl", "all"])

    sub.add_parser("status", help="Inspect available local model artifacts")

    args = parser.parse_args(argv)
    cmd = args.command or "demo"

    if cmd == "status":
        print(json.dumps(ModelRegistry().status(), indent=2, ensure_ascii=False))
        return 0

    if cmd == "download":
        targets = ["ml", "dl"] if args.model == "all" else [args.model]
        for kind in targets:
            existing = find_artifact(kind)
            if existing is not None:
                print(f"[download] {kind} already present: {existing}")
                continue
            print(f"[download] fetching {kind} from {DRIVE_FOLDER_URL} ...")
            out = download_from_drive(kind)
            print(f"[download] saved {out} ({out.stat().st_size / (1024 ** 2):.2f} MB)")
        return 0

    if cmd == "predict":
        t1 = _read_text_arg(args.text_1, args.text_file_1)
        t2 = _read_text_arg(args.text_2, args.text_file_2)
        reg = ModelRegistry(device=args.device)
        res = reg.predict(args.model, t1, t2, check_symmetry=args.check_symmetry)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    serve_demo(
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", 8000),
        open_browser=not getattr(args, "no_browser", False),
        device=getattr(args, "device", None),
        preload=getattr(args, "preload", None),
    )
    return 0
