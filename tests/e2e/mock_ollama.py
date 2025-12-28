import hashlib
import json
import random
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List


MODEL_DIMS: Dict[str, int] = {
    "snowflake-arctic-embed:22m": 384,
    "nomic-embed-text:v1.5": 768,
    "embeddinggemma:300m": 768,
    "snowflake-arctic-embed2:568m": 1024,
    "qwen3-embedding:4b": 2560,
}


def _deterministic_vector(model: str, text: str, dims: int) -> List[float]:
    seed_material = hashlib.sha256(f"{model}|{text}".encode("utf-8")).digest()
    seed_int = int.from_bytes(seed_material[:8], "big")
    rng = random.Random(seed_int)
    return [rng.uniform(-1.0, 1.0) for _ in range(dims)]


def _create_handler():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/api/embed":
                self.send_error(404, "Not Found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            model = payload.get("model")
            inputs = payload.get("input")

            if not isinstance(model, str):
                self.send_error(400, "model is required")
                return

            dims = MODEL_DIMS.get(model)
            if dims is None:
                self.send_error(400, f"unknown model: {model}")
                return

            if isinstance(inputs, str):
                texts = [inputs]
            elif isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
                texts = inputs
            else:
                self.send_error(400, "input must be string or list of strings")
                return

            embeddings = [_deterministic_vector(model, text, dims) for text in texts]

            body = json.dumps({"model": model, "embeddings": embeddings}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # pragma: no cover - keep test noise low
            return

    return Handler


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class MockOllamaServer:
    def __init__(self) -> None:
        self.port = _find_free_port()
        self._server = HTTPServer(("127.0.0.1", self.port), _create_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
