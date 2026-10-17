"""HTTP layer. Standard library only: s390x has no flask wheel and no pyarrow.

Routes (contract in docs/implementation-plan.md §3):
    GET  /                    page shell
    GET  /static/<file>       assets
    GET  /api/matches         precomputed list
    GET  /api/series/<uid>    precomputed panel + alerts
    POST /api/score           live inference (the s390x proof)
    GET  /api/health          arch, python, models, uptime

Acceptance: /api/health returns "arch": "s390x" (on the mainframe) and
/api/score returns a probability plus a latency in single-digit ms.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

SITE_DIR = Path(os.environ.get("CS2_SITE_DIR", "site"))
DATA_DIR = SITE_DIR / "data"
START = time.time()

STATE: dict = {"matches": [], "series": {}, "summary": {}, "model_error": None}

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def load_artifacts() -> None:
    try:
        STATE["matches"] = json.loads((DATA_DIR / "matches.json").read_text())
        STATE["summary"] = json.loads((DATA_DIR / "summary.json").read_text())
        for m in STATE["matches"]:
            p = DATA_DIR / "series" / f"{m['map_uid']}.json"
            if p.exists():
                STATE["series"][m["map_uid"]] = json.loads(p.read_text())
        print(f"loaded {len(STATE['matches'])} maps, {len(STATE['series'])} series from {DATA_DIR}")
    except FileNotFoundError as exc:
        print(f"WARNING: artifacts missing ({exc}); run `python -m pipeline.build_artifacts`")
    try:
        from models import score as score_mod
        score_mod.load_models(refresh=True)
    except Exception as exc:  # noqa: BLE001
        STATE["model_error"] = f"{type(exc).__name__}: {exc}"
        print(f"WARNING: model unavailable ({STATE['model_error']}); /api/score will 503")


class Handler(BaseHTTPRequestHandler):
    server_version = "cs2-integrity/1.0"

    def log_message(self, fmt: str, *args) -> None:  # keep the console readable
        print(f"{self.address_string()} {fmt % args}")

    # ---------- helpers ----------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Robots-Tag", "noindex, nofollow")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            # never echo the server's filesystem layout to a client
            self._json(404, {"error": "not found", "path": "/" + path.name})
            return
        ctype = MIME.get(path.suffix, "application/octet-stream")
        self._send(200, path.read_bytes(), ctype)

    def _static(self, rel: str, base: Path) -> None:
        """Serve rel under base, or 403. Containment is checked on the path
        components, not on a string prefix: `startswith` would also accept a
        sibling directory named `static_evil`."""
        base = base.resolve()
        target = (base / rel).resolve()
        if target != base and base not in target.parents:
            self._json(403, {"error": "forbidden"})
            return
        self._file(target)

    # ---------- routes ----------
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._file(SITE_DIR / "index.html")
        elif path.startswith("/static/"):
            self._static(path[len("/static/"):], SITE_DIR / "static")
        elif path.startswith("/data/"):
            # the static fallback bundle the page uses when the API is down
            self._static(path[len("/data/"):], SITE_DIR / "data")
        elif path == "/api/matches":
            if not STATE["matches"]:
                self._json(503, {"error": "artifacts not built",
                                 "hint": "python -m pipeline.build_artifacts --out site/data"})
                return
            self._json(200, STATE["matches"])
        elif path.startswith("/api/series/"):
            uid = path[len("/api/series/"):]
            doc = STATE["series"].get(uid)
            if doc is None:
                self._json(404, {"error": f"no series for {uid}", "map_uid": uid})
                return
            self._json(200, doc)
        elif path == "/api/health":
            self._json(200, {
                "ok": STATE["model_error"] is None,
                "arch": platform.machine(),
                "uname": platform.uname().system + " " + platform.uname().machine,
                "python": platform.python_version(),
                "hostname": socket.gethostname(),
                "models_loaded": [] if STATE["model_error"] else ["logit-v1", "xgb-v1"],
                "model_error": STATE["model_error"],
                "n_maps": len(STATE["matches"]),
                "uptime_s": round(time.time() - START, 1),
                "resolution_floor_s": STATE["summary"].get("resolution_floor_s", 60),
            })
        elif path == "/api/summary":
            self._json(200, STATE["summary"] or {"error": "not built"})
        else:
            self._json(404, {"error": "unknown route", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/score":
            self._json(404, {"error": "unknown route", "path": path})
            return
        if STATE["model_error"]:
            self._json(503, {"error": "model unavailable", "detail": STATE["model_error"]})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}").get("round_state", {})
        except (ValueError, AttributeError) as exc:
            self._json(400, {"error": f"bad request body: {exc}"})
            return
        try:
            from models import score as score_mod
            self._json(200, score_mod.score_round_state(body))
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def serve(host: str = "0.0.0.0", port: int = 38888) -> None:
    load_artifacts()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"serving {SITE_DIR} on http://{host}:{port}  "
          f"(arch={platform.machine()}, python={platform.python_version()})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "38888")))
    args = ap.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
