"""Local HTTP server for the MIT Red Authorship Verification web demo (ML + DL selector)."""
from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from nlu_authorship.artifacts import DRIVE_FOLDER_URL, find_artifact
from nlu_authorship.presets import DEMO_PRESETS

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ModelRegistry:
    """Thread-safe lazy loader for Track 01 (Stylometric ML) and Track 03 (DeBERTa-v3-large)."""

    def __init__(self, device: str | None = None):
        self._lock = threading.Lock()
        self._ml = None
        self._dl = None
        self._device = device

    def status(self) -> dict[str, Any]:
        ml_path = find_artifact("ml")
        dl_path = find_artifact("dl")
        return {
            "ml": {
                "available": ml_path is not None,
                "loaded": self._ml is not None,
                "path": str(ml_path) if ml_path else None,
                "name": "01. Stylometric ML (10-Expert + OOD Gate)",
                "runtime": "CPU (~32 ms/pair, 21.6 MB)",
            },
            "dl": {
                "available": dl_path is not None,
                "loaded": self._dl is not None,
                "path": str(dl_path) if dl_path else None,
                "name": "03. DeBERTa-v3-large (Dual-ABTT + Soup)",
                "runtime": "GPU/CPU (437M params, 860 MB)",
            },
            "drive_url": DRIVE_FOLDER_URL,
        }

    def get_ml(self):
        with self._lock:
            if self._ml is None:
                from nlu_authorship.ml_engine import StylometricMLEngine
                self._ml = StylometricMLEngine()
            return self._ml

    def get_dl(self):
        with self._lock:
            if self._dl is None:
                from nlu_authorship.dl_engine import DebertaLargeEngine
                self._dl = DebertaLargeEngine(device=self._device)
            return self._dl

    def predict(self, model_mode: str, text_1: str, text_2: str, check_symmetry: bool = False) -> dict[str, Any]:
        t1, t2 = text_1.strip(), text_2.strip()
        if not t1 or not t2:
            raise ValueError("Both Passage A and Passage B must be non-empty.")

        results: dict[str, Any] = {}
        if model_mode in ("ml", "both"):
            results["ml"] = self.get_ml().predict_pair(t1, t2, check_symmetry=check_symmetry)
        if model_mode in ("dl", "both"):
            results["dl"] = self.get_dl().predict_pair(t1, t2, check_symmetry=check_symmetry)
        if not results:
            raise ValueError(f"Unsupported model mode: {model_mode}")
        return {"mode": model_mode, "results": results}


def create_server(host: str = "127.0.0.1", port: int = 8000, device: str | None = None, preload: str | None = None) -> ThreadingHTTPServer:
    registry = ModelRegistry(device=device)
    if preload in ("ml", "both") and find_artifact("ml") is not None:
        registry.get_ml()
    if preload in ("dl", "both") and find_artifact("dl") is not None:
        registry.get_dl()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                html = (STATIC_DIR / "index.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if path == "/api/status":
                self._send_json(HTTPStatus.OK, registry.status())
                return
            if path == "/api/presets":
                self._send_json(HTTPStatus.OK, {"presets": DEMO_PRESETS})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/api/predict":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 200_000:
                    raise ValueError("Invalid request body size.")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                out = registry.predict(
                    model_mode=str(body.get("model", "ml")),
                    text_1=str(body.get("text_1", "")),
                    text_2=str(body.get("text_2", "")),
                    check_symmetry=bool(body.get("check_symmetry", False)),
                )
                self._send_json(HTTPStatus.OK, out)
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    return ThreadingHTTPServer((host, port), Handler)


def serve_demo(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True, device: str | None = None, preload: str | None = None) -> None:
    server = create_server(host=host, port=port, device=device, preload=preload)
    url = f"http://{host}:{port}/"
    print(f"[demo] Authorship Verification Web UI listening at {url}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] shutting down")
    finally:
        server.server_close()
