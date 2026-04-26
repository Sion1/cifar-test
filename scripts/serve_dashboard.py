#!/usr/bin/env python3
"""Serve the generated dashboard and persist user summaries.

Usage:
  python3 scripts/serve_dashboard.py

Then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "docs" / "autoresearch_dashboard"
SUMMARY_PATH = ROOT / "state" / "user_summary.md"
NODE_SUMMARY_PATH = ROOT / "state" / "user_summaries.md"


def read_node_summaries() -> dict[str, str]:
    if not NODE_SUMMARY_PATH.exists():
        return {}
    text = NODE_SUMMARY_PATH.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    parts = re.split(r"(?m)^<!--\s*node:([^>]+?)\s*-->\s*$", text)
    for i in range(1, len(parts), 2):
        node_id = parts[i].strip()
        body = re.sub(r"(?m)^##\s+.*?\n", "", parts[i + 1], count=1).strip()
        if node_id:
            out[node_id] = body
    return out


def write_node_summaries(items: dict[str, str], names: dict[str, str] | None = None) -> None:
    names = names or {}
    lines = [
        "# User Node Summaries",
        "",
        "This file is written by the dashboard. AutoResearch agents should read it before choosing or analyzing experiments.",
        "",
    ]
    for node_id in sorted(items):
        text = items[node_id].rstrip()
        if not text:
            continue
        title = names.get(node_id) or node_id
        lines += [f"<!-- node:{node_id} -->", f"## {title}", "", text, ""]
    NODE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    NODE_SUMMARY_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "AutoResearchDashboard/1.0"

    def _send(self, code: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(self.path.split("?", 1)[0])
        if path == "/api/user-summary":
            text = SUMMARY_PATH.read_text(encoding="utf-8", errors="replace") if SUMMARY_PATH.exists() else ""
            self._send(200, json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path == "/api/node-summary":
            self._send(200, json.dumps({"items": read_node_summaries()}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path in {"", "/"}:
            rel = "index.html"
        else:
            rel = path.lstrip("/")
        target = (DASHBOARD_DIR / rel).resolve()
        try:
            target.relative_to(DASHBOARD_DIR.resolve())
        except ValueError:
            self._send(403, b"forbidden")
            return
        if not target.exists() or not target.is_file():
            self._send(404, b"not found")
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self) -> None:  # noqa: N802
        api_path = self.path.split("?", 1)[0]
        if api_path not in {"/api/user-summary", "/api/node-summary"}:
            self._send(404, b"not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            text = str(payload.get("text", ""))
        except Exception:
            self._send(400, b"bad json")
            return
        if api_path == "/api/user-summary":
            SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
            SUMMARY_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")
            self._send(200, json.dumps({"ok": True, "path": str(SUMMARY_PATH)}).encode("utf-8"), "application/json")
            return
        node_id = str(payload.get("node_id", "")).strip()
        node_name = str(payload.get("node_name", node_id)).strip()
        if not node_id:
            self._send(400, b"missing node_id")
            return
        items = read_node_summaries()
        items[node_id] = text
        names = {node_id: node_name}
        write_node_summaries(items, names)
        self._send(200, json.dumps({"ok": True, "path": str(NODE_SUMMARY_PATH)}).encode("utf-8"), "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not (DASHBOARD_DIR / "index.html").exists():
        raise SystemExit("Dashboard not generated. Run: bash scripts/generate_experiment_tree_web.sh")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving dashboard: http://{args.host}:{args.port}/")
    print(f"Global summary path: {SUMMARY_PATH}")
    print(f"Node summaries path: {NODE_SUMMARY_PATH}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
