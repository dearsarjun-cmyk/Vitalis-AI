"""VITALIS public/local web server.

Local:  uv run python web_app.py
Deploy: bind to 0.0.0.0 and honor the platform's PORT environment variable.
"""
from __future__ import annotations

import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from tokenizers import Tokenizer

from hello import Brain, CHECKPOINT_PATH, TOKENIZER_PATH, generate

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))
_model_lock = threading.Lock()
_model = None
_tokenizer = None
_model_error = None


def load_model():
    global _model, _tokenizer, _model_error
    with _model_lock:
        if _model is not None:
            return True
        if not TOKENIZER_PATH.exists() or not CHECKPOINT_PATH.exists():
            _model_error = "No trained checkpoint found."
            return False
        try:
            saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
            _model = Brain(**saved["config"])
            _model.load_state_dict(saved["model_state_dict"])
            _model.eval()
            _model_error = None
            return True
        except Exception as exc:
            _model_error = f"Could not load checkpoint: {exc}"
            _model = None
            _tokenizer = None
            return False


class Handler(BaseHTTPRequestHandler):
    server_version = "VITALIS/1.0"

    def _send(self, status: int, body: bytes, content_type: str, *, no_store: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; script-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'self'; frame-ancestors 'none'")
        self.send_header("Cache-Control", "no-store" if no_store else "public, max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", no_store=True)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/" or path == "/index.html":
            file_path = STATIC_DIR / "index.html"
        elif path.startswith("/static/"):
            relative = path.removeprefix("/static/")
            # Only serve files directly inside the static directory.
            if not relative or "/" in relative or ".." in relative or "\\" in relative:
                self._json(404, {"error": "Not found"})
                return
            file_path = STATIC_DIR / relative
        elif path == "/status" or path == "/health":
            ready = load_model()
            self._json(200, {
                "ready": ready,
                "model": "VITALIS Core",
                "message": "Ready" if ready else (_model_error or "Model is not trained yet."),
                "training_required": not ready,
            })
            return
        else:
            self._json(404, {"error": "Not found"})
            return

        try:
            body = file_path.read_bytes()
        except OSError:
            self._json(404, {"error": "Not found"})
            return
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript",):
            content_type += "; charset=utf-8"
        self._send(200, body, content_type)

    def do_POST(self):
        if urlparse(self.path).path != "/chat":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20_000:
                self._json(413, {"error": "Message too long or empty."})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                self._json(400, {"error": "Enter a message first."})
                return
            if len(prompt) > 4000:
                self._json(400, {"error": "Please keep messages under 4,000 characters."})
                return
            if not load_model():
                self._json(503, {"error": "VITALIS is online, but no trained model checkpoint is installed yet. Train the model and deploy vitalis_model.pt plus tokenizer.json."})
                return
            # One generation at a time protects CPU/RAM on small hosting instances.
            with _model_lock, torch.inference_mode():
                answer = generate(_model, _tokenizer, prompt, max_new_tokens=120, temperature=0.8, top_k=40)
            self._json(200, {"reply": answer.strip() or "The model generated no text. Try a different prompt."})
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json(400, {"error": f"Invalid request: {exc}"})
        except Exception as exc:
            print(f"[VITALIS] generation failed: {exc}")
            self._json(500, {"error": "Generation failed. Check the server logs for details."})

    def log_message(self, fmt, *args):
        print("[web] " + (fmt % args))


def main():
    print("VITALIS AI website starting…")
    print(f"Listening on {HOST}:{PORT}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VITALIS web server…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
