"""Serveur de monitoring stdlib (pas de dependance externe).

Usage :
    uv run python -m monitor.server --run runs/encoder_v1 --port 8765
    uv run python -m monitor.server --follow --port 8765   # suit le run actif de la boucle

Endpoints :
  GET  /              -> dashboard.html (rendu inline avec le path du run injecte)
  GET  /history.json  -> contenu live de runs/<run>/history.json
  GET  /control.json  -> contenu live du control.json
  GET  /config.json   -> contenu du config.json
  GET  /heartbeat.json-> contenu du heartbeat.json
  GET  /info          -> {"run": "<dossier actif>"}
  POST /control       -> body JSON, merge dans control.json
                         {"pause": true} | {"stop": true} | {"checkpoint_now": true} | ...

Mode --follow : le run actif n'est pas fige. A chaque requete le serveur choisit le
dossier dont le heartbeat.json a ete modifie le plus recemment parmi --follow-glob
(par defaut les runs de la boucle DPO). Le dashboard suit ainsi la chaine
cNN_sft -> dpoloop_cNN (campagne) -> cNN_dpo -> c(N+1)_sft sans redemarrage.

Le serveur relit TOUS les fichiers a chaque requete, y compris dashboard.html
(permet d'iterer sur le front sans redemarrer le serveur). Suffit largement au
volume d'un dashboard (1 poll / 2s).
"""
from __future__ import annotations
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .control import init_control_file, ControlState


def make_resolver(follow_globs, fallback: Path):
    """Retourne une fonction resolve() -> Path du run actif.

    Si follow_globs est fourni, le run actif = le dossier dont le heartbeat.json a
    le mtime le plus recent (le training/gen en cours ecrit son heartbeat en continu,
    donc c'est toujours le plus frais). Sinon, run fixe = fallback."""
    if not follow_globs:
        return lambda: fallback

    def resolve() -> Path:
        best, best_m = None, -1.0
        for g in follow_globs:
            for hb in Path(".").glob(g.rstrip("/") + "/heartbeat.json"):
                try:
                    m = hb.stat().st_mtime
                except OSError:
                    continue
                if m > best_m:
                    best_m, best = m, hb.parent
        return best or fallback

    return resolve


def make_handler(resolve_run, dashboard_path: Path, dashboard_fallback: str):
    inited: set[str] = set()

    def control_path_for(run_dir: Path) -> Path:
        key = str(run_dir)
        if key not in inited:
            init_control_file(run_dir)   # reset propre a la 1ere vue d'un run
            inited.add(key)
        return run_dir / "control.json"

    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # silence
            pass

        def _send_json(self, code: int, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file_json(self, path: Path):
            if not path.exists():
                return self._send_json(200, [] if path.name == "history.json" else {})
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            run_dir = resolve_run()
            if self.path in ("/", "/index.html"):
                # Re-lecture a chaque requete : permet d'iterer sur dashboard.html
                # sans redemarrer le serveur. Fallback sur le contenu charge au
                # demarrage si le fichier devient illisible.
                try:
                    html = dashboard_path.read_text(encoding="utf-8")
                except Exception:
                    html = dashboard_fallback
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/history.json":
                return self._send_file_json(run_dir / "history.json")
            if self.path == "/control.json":
                return self._send_file_json(control_path_for(run_dir))
            if self.path == "/config.json":
                return self._send_file_json(run_dir / "config.json")
            if self.path == "/heartbeat.json":
                return self._send_file_json(run_dir / "heartbeat.json")
            if self.path == "/info":
                return self._send_json(200, {"run": str(run_dir)})
            self.send_response(404); self.end_headers()

        def do_POST(self):
            if self.path != "/control":
                self.send_response(404); self.end_headers(); return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                patch = json.loads(body)
            except Exception:
                return self._send_json(400, {"error": "invalid json"})

            control_path = control_path_for(resolve_run())   # cible le run actif
            state = ControlState.load(control_path)
            for k in ("pause", "stop", "checkpoint_now"):
                if k in patch:
                    setattr(state, k, bool(patch[k]))
            state.save(control_path)
            return self._send_json(200, {"ok": True, **{k: getattr(state, k) for k in ("pause", "stop", "checkpoint_now")}})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None,
                    help="Dossier du run fixe (ex: runs/encoder_v1). Ignore si --follow.")
    ap.add_argument("--follow", action="store_true",
                    help="Suit le run actif de la boucle (heartbeat le plus recent).")
    ap.add_argument("--follow-glob", action="append", default=None,
                    help="Glob(s) des runs a suivre. Defaut : boucle DPO. Repetable.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dashboard", type=Path,
                    default=Path(__file__).resolve().parents[2] / "dashboard.html",
                    help="Chemin vers dashboard.html")
    args = ap.parse_args()

    if args.follow:
        globs = args.follow_glob or ["runs/dpo_loop/*", "runs/arena/dpoloop_*"]
        fallback = args.run or Path("runs/dpo_loop")
        resolve_run = make_resolver(globs, fallback)
        label = "follow " + " ".join(globs)
    else:
        if args.run is None:
            ap.error("--run est requis sans --follow")
        args.run.mkdir(parents=True, exist_ok=True)
        resolve_run = make_resolver(None, args.run)
        label = str(args.run)

    html = args.dashboard.read_text(encoding="utf-8")  # fallback initial
    Handler = make_handler(resolve_run, args.dashboard, html)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[monitor] serving http://127.0.0.1:{args.port}/ ({label})")
    print(f"[monitor] active run: {resolve_run()}")
    print(f"[monitor] Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
