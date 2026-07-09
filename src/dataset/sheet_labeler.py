"""Serveur du labeler de sprite-sheets (l'humain groupe en cycles + tague).

Plomberie deterministe : segmente le sheet (sheet_segment, scipy), sert le PNG + les
bounding-boxes, et l'UI HTML. L'utilisateur assemble les frames en cycles et choisit
action/direction ; le POST /save ecrit un manifeste JSON que l'ingestion relit. Claude
n'intervient jamais dans le jugement visuel (cf memoire dataset-enrichment).

Usage :
    set PYTHONPATH=src
    uv run python -m dataset.sheet_labeler --sheet runs/tsr/link_cap.png --out data/labels/link_cap.json --port 8767
"""
from __future__ import annotations
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .sheet_segment import segment_sheet, cluster_rows

HTML = Path(__file__).resolve().parents[2] / "sheet_labeler.html"


def build_manifest_stub(sheet: Path, boxes) -> dict:
    rows = cluster_rows(boxes)
    return {
        "sheet": str(sheet),
        "rows": [[[b.y0, b.x0, b.y1, b.x1] for b in row] for row in rows],
    }


def make_handler(sheet: Path, out: Path, dilate: int, min_px: int, max_px: int):
    boxes = segment_sheet(sheet, dilate=dilate, min_px=min_px, max_px=max_px)
    manifest = build_manifest_stub(sheet, boxes)
    sheet_bytes = sheet.read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = HTML.read_text(encoding="utf-8")
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if self.path == "/sheet.png":
                return self._send(200, sheet_bytes, "image/png")
            if self.path == "/boxes.json":
                return self._send(200, json.dumps(manifest).encode("utf-8"), "application/json")
            if self.path == "/existing.json":
                data = out.read_bytes() if out.exists() else b"null"
                return self._send(200, data, "application/json")
            self.send_response(404); self.end_headers()

        def do_POST(self):
            if self.path != "/save":
                self.send_response(404); self.end_headers(); return
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode("utf-8") if n else "{}"
            try:
                payload = json.loads(body)
            except Exception:
                return self._send(400, b'{"error":"bad json"}', "application/json")
            out.parent.mkdir(parents=True, exist_ok=True)
            payload["sheet"] = str(sheet)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            ncyc = len(payload.get("cycles", []))
            return self._send(200, json.dumps({"ok": True, "cycles": ncyc, "path": str(out)}).encode(), "application/json")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="manifeste JSON de sortie")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--min-px", type=int, default=8)
    ap.add_argument("--max-px", type=int, default=40)
    args = ap.parse_args()

    Handler = make_handler(args.sheet, args.out, args.dilate, args.min_px, args.max_px)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[labeler] http://127.0.0.1:{args.port}/  sheet={args.sheet}  -> {args.out}")
    print("[labeler] Ctrl+C pour arreter")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[labeler] stop")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
