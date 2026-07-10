"""Serveur du labeler de sprite-sheets (mode dossier : l'humain choisit la feuille).

Plomberie deterministe : segmente chaque feuille (sheet_segment, scipy), sert le PNG +
les bounding-boxes, et l'UI HTML avec un selecteur de feuille. L'utilisateur assemble les
frames en cycles et tague action/direction ; POST /save ecrit un manifeste JSON PAR FEUILLE
(<out-dir>/<nom>.json) que l'ingestion relit. Claude n'intervient jamais dans le jugement
visuel (cf memoire dataset-enrichment).

Usage :
    set PYTHONPATH=src
    uv run python -m dataset.sheet_labeler --sheets-dir runs/tsr --out-dir data/labels --port 8767
"""
from __future__ import annotations
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .sheet_segment import segment_sheet, cluster_rows

HTML = Path(__file__).resolve().parents[2] / "sheet_labeler.html"


def build_manifest_stub(sheet: Path, boxes) -> dict:
    rows = cluster_rows(boxes)
    return {"sheet": str(sheet),
            "rows": [[[b.y0, b.x0, b.y1, b.x1] for b in row] for row in rows]}


def make_handler(sheets_dir: Path, out_dir: Path, seg: dict):
    sheets = sorted(p.name for p in sheets_dir.glob("*.png"))
    png_cache: dict[str, bytes] = {}
    seg_cache: dict[tuple, dict] = {}   # (name, dilate, min_px, max_px) -> manifest

    def label_path(name: str) -> Path:
        return out_dir / (Path(name).stem + ".json")

    def png(name: str) -> bytes:
        if name not in png_cache:
            png_cache[name] = (sheets_dir / name).read_bytes()
        return png_cache[name]

    def manifest(name: str, dilate: int, min_px: int, max_px: int) -> dict:
        key = (name, dilate, min_px, max_px)
        if key not in seg_cache:
            path = sheets_dir / name
            boxes = segment_sheet(path, dilate=dilate, min_px=min_px, max_px=max_px)
            seg_cache[key] = build_manifest_stub(path, boxes)
        return seg_cache[key]

    def qint(q, k, d):
        try:
            return int(q.get(k, [d])[0])
        except (TypeError, ValueError):
            return d

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

        def _name(self):
            n = parse_qs(urlparse(self.path).query).get("name", [None])[0]
            return n if n in sheets else None

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, HTML.read_text(encoding="utf-8").encode("utf-8"), "text/html; charset=utf-8")
            if path == "/sheets.json":
                lst = [{"name": s, "labeled": label_path(s).exists()} for s in sheets]
                return self._send(200, json.dumps(lst).encode("utf-8"), "application/json")
            name = self._name()
            if path == "/sheet.png" and name:
                return self._send(200, png(name), "image/png")
            if path == "/boxes.json" and name:
                q = parse_qs(urlparse(self.path).query)
                m = manifest(name, qint(q, "dilate", seg["dilate"]),
                             seg["min_px"], qint(q, "max_px", seg["max_px"]))
                return self._send(200, json.dumps(m).encode("utf-8"), "application/json")
            if path == "/existing.json" and name:
                p = label_path(name)
                return self._send(200, p.read_bytes() if p.exists() else b"null", "application/json")
            self.send_response(404); self.end_headers()

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                self.send_response(404); self.end_headers(); return
            name = self._name()
            if not name:
                return self._send(400, b'{"error":"unknown sheet"}', "application/json")
            n = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(n).decode("utf-8") if n else "{}")
            except Exception:
                return self._send(400, b'{"error":"bad json"}', "application/json")
            out_dir.mkdir(parents=True, exist_ok=True)
            payload["sheet"] = str(sheets_dir / name)
            label_path(name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            ncyc = len(payload.get("cycles", []))
            return self._send(200, json.dumps({"ok": True, "cycles": ncyc, "path": str(label_path(name))}).encode(), "application/json")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets-dir", type=Path, required=True, help="dossier de feuilles *.png")
    ap.add_argument("--out-dir", type=Path, required=True, help="dossier des manifestes JSON (1 par feuille)")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--min-px", type=int, default=8)
    ap.add_argument("--max-px", type=int, default=64)
    args = ap.parse_args()

    seg = {"dilate": args.dilate, "min_px": args.min_px, "max_px": args.max_px}
    Handler = make_handler(args.sheets_dir, args.out_dir, seg)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    n = len(sorted(args.sheets_dir.glob("*.png")))
    print(f"[labeler] http://127.0.0.1:{args.port}/  {n} feuilles dans {args.sheets_dir} -> {args.out_dir}")
    print("[labeler] Ctrl+C pour arreter")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[labeler] stop")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
