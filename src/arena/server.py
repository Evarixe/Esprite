"""Serveur de l'arène d'éval humaine (stdlib, même pattern que monitor.server).

Sert arena.html + une API JSON sur une campagne pré-générée (cf arena.campaign).
Aucun GPU requis. Un seul juge local — pas d'auth, pas de concurrence d'écriture
(verrou process sur votes.jsonl).

Endpoints :
  GET  /                    -> arena.html (relu à chaque requête, itération front libre)
  GET  /campaign.json       -> campagne (items avec ref+palette, checkpoints)
  GET  /gen?ckpt=ID&item=ID -> génération stockée d'un item pour un checkpoint
  GET  /next_matchup        -> prochain duel (couverture équilibrée, côtés randomisés)
                               ?exclude=id1,id2 pour retirer des compétiteurs
  GET  /votes.json          -> tous les votes (pour l'UI : progression, planches déjà notées)
  GET  /results.json        -> classement Bradley-Terry global + par axe + breakdowns
  POST /vote                -> enregistre un vote (arena ou plank), append votes.jsonl
  POST /undo                -> retire le dernier vote

Usage :
    set PYTHONPATH=src
    uv run python -m arena.server --campaign runs/arena/20k_vs_15k --port 8766
"""
from __future__ import annotations
import argparse
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .ranking import rank_axis

AXES = ("global", "movement", "fidelity", "cleanliness")


def _sanitize(o):
    """Remplace NaN/Inf par None (JSON valide) — les CI bootstrap sur peu de duels
    produisent des np.nan que json.dumps sort en `NaN` littéral, illégal côté JS."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(x) for x in o]
    return o


class VoteStore:
    """votes.jsonl append-only + état en mémoire. Verrou process (juge unique)."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.votes: list[dict] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self.votes.append(json.loads(line))

    def append(self, vote: dict) -> None:
        with self.lock:
            self.votes.append(vote)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(vote, ensure_ascii=False) + "\n")

    def undo(self) -> dict | None:
        with self.lock:
            if not self.votes:
                return None
            removed = self.votes.pop()
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for v in self.votes:
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
            tmp.replace(self.path)   # même pattern atomique que le heartbeat
            return removed


def outcome_to_score(outcome: str) -> float | None:
    """'left' -> 1.0 (gauche gagne), 'right' -> 0.0, 'tie'/'both_bad' -> 0.5."""
    return {"left": 1.0, "right": 0.0, "tie": 0.5, "both_bad": 0.5}.get(outcome)


def compute_results(campaign: dict, votes: list[dict]) -> dict:
    ids = [c["id"] for c in campaign["checkpoints"]]
    idx = {cid: k for k, cid in enumerate(ids)}
    items_by_id = {it["item_id"]: it for it in campaign["items"]}

    arena_votes = [v for v in votes if v.get("mode") == "arena"]
    duels_by_axis: dict[str, list[tuple[int, int, float]]] = {a: [] for a in AXES}
    both_bad_count: dict[str, int] = {cid: 0 for cid in ids}
    per_slice: dict[str, dict[str, list]] = {}   # slice -> axis 'global' duels

    for v in arena_votes:
        li, ri = idx.get(v["left"]), idx.get(v["right"])
        if li is None or ri is None:
            continue
        for axis in AXES:
            o = v.get("axes", {}).get(axis) if axis != "global" else v.get("outcome")
            s = outcome_to_score(o) if o else None
            if s is not None:
                duels_by_axis[axis].append((li, ri, s))
        if v.get("outcome") == "both_bad":
            both_bad_count[v["left"]] += 1
            both_bad_count[v["right"]] += 1
        it = items_by_id.get(v.get("item_id"))
        if it is not None and v.get("outcome"):
            s = outcome_to_score(v["outcome"])
            if s is not None:
                for sl in (f"action:{it['action']}",
                           f"N:{it['n_frames']}",
                           f"ref:{'with' if it['with_ref'] else 'no'}"):
                    per_slice.setdefault(sl, []).append((li, ri, s))

    results = {
        "n_votes": len(votes),
        "n_arena": len(arena_votes),
        "axes": {a: rank_axis(ids, duels_by_axis[a]) for a in AXES},
        "slices": {sl: rank_axis(ids, d, with_ci=False) for sl, d in sorted(per_slice.items())},
        "both_bad": both_bad_count,
    }

    # Planches : moyenne des notes 1-5 + taux de flags par checkpoint
    plank_votes = [v for v in votes if v.get("mode") == "plank"]
    plank: dict[str, dict] = {}
    for v in plank_votes:
        cid = v.get("ckpt")
        if cid not in idx:
            continue
        d = plank.setdefault(cid, {"ratings": [], "flags": 0, "n": 0})
        d["n"] += 1
        if v.get("rating") is not None:
            d["ratings"].append(float(v["rating"]))
        if v.get("flag"):
            d["flags"] += 1
    results["plank"] = {
        cid: {
            "n_votes": d["n"],
            "mean_rating": round(sum(d["ratings"]) / len(d["ratings"]), 2) if d["ratings"] else None,
            "n_flags": d["flags"],
        } for cid, d in plank.items()
    }
    return results


def next_matchup(campaign: dict, votes: list[dict], exclude: set[str],
                 rng: random.Random) -> dict | None:
    """Couverture équilibrée : (paire, item) le moins voté, côté randomisé."""
    ids = [c["id"] for c in campaign["checkpoints"] if c["id"] not in exclude]
    if len(ids) < 2:
        return None
    counts: dict[tuple[str, str, str], int] = {}
    for v in votes:
        if v.get("mode") != "arena":
            continue
        pair = tuple(sorted((v["left"], v["right"])))
        counts[(pair[0], pair[1], v["item_id"])] = \
            counts.get((pair[0], pair[1], v["item_id"]), 0) + 1
    candidates = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pair = tuple(sorted((ids[a], ids[b])))
            for it in campaign["items"]:
                c = counts.get((pair[0], pair[1], it["item_id"]), 0)
                candidates.append((c, pair[0], pair[1], it["item_id"]))
    if not candidates:
        return None
    min_c = min(c[0] for c in candidates)
    pool = [c for c in candidates if c[0] == min_c]
    _, x, y, item_id = rng.choice(pool)
    left, right = (x, y) if rng.random() < 0.5 else (y, x)
    total_pairs = len(ids) * (len(ids) - 1) // 2 * len(campaign["items"])
    return {"item_id": item_id, "left": left, "right": right,
            "coverage": {"min_votes_per_cell": min_c, "total_cells": total_pairs,
                         "voted_cells": len(counts)}}


def make_handler(campaign_dir: Path, html_path: Path, store: VoteStore, target_duels: int = 0):
    rng = random.Random()

    def load_campaign() -> dict:
        c = json.loads((campaign_dir / "campaign.json").read_text(encoding="utf-8"))
        c["target_duels"] = target_duels   # quota de la passe d'éval (0 = illimité)
        return c

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence
            pass

        def _send_json(self, code: int, obj) -> None:
            body = json.dumps(_sanitize(obj), ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                html = html_path.read_text(encoding="utf-8")
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if url.path == "/campaign.json":
                return self._send_json(200, load_campaign())
            if url.path == "/gen":
                ckpt = q.get("ckpt", [""])[0]
                item = q.get("item", [""])[0]
                p = campaign_dir / "gens" / ckpt / f"{item}.json"
                if not p.exists() or ".." in ckpt or ".." in item:
                    return self._send_json(404, {"error": f"gen introuvable: {ckpt}/{item}"})
                return self._send_json(200, json.loads(p.read_text(encoding="utf-8")))
            if url.path == "/next_matchup":
                exclude = set(filter(None, q.get("exclude", [""])[0].split(",")))
                m = next_matchup(load_campaign(), store.votes, exclude, rng)
                if m is None:
                    return self._send_json(200, {"error": "moins de 2 compétiteurs"})
                return self._send_json(200, m)
            if url.path == "/votes.json":
                return self._send_json(200, store.votes)
            if url.path == "/results.json":
                return self._send_json(200, compute_results(load_campaign(), store.votes))
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body)
            except Exception:
                return self._send_json(400, {"error": "invalid json"})

            if self.path == "/vote":
                mode = payload.get("mode")
                if mode == "arena":
                    required = ("item_id", "left", "right")
                elif mode == "plank":
                    required = ("ckpt", "item_id")
                else:
                    return self._send_json(400, {"error": f"mode inconnu: {mode}"})
                missing = [k for k in required if not payload.get(k)]
                if missing:
                    return self._send_json(400, {"error": f"champs manquants: {missing}"})
                # Vote arène : outcome (global) OU au moins un axe coché.
                if mode == "arena" and not payload.get("outcome") and not payload.get("axes"):
                    return self._send_json(400, {"error": "vote arène : outcome ou axes requis"})
                payload["ts"] = time.time()
                store.append(payload)
                return self._send_json(200, {"ok": True, "n_votes": len(store.votes)})

            if self.path == "/undo":
                removed = store.undo()
                if removed is None:
                    return self._send_json(200, {"ok": False, "error": "aucun vote"})
                return self._send_json(200, {"ok": True, "removed": removed,
                                             "n_votes": len(store.votes)})
            self.send_response(404)
            self.end_headers()

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True,
                    help="Dossier de campagne (ex: runs/arena/20k_vs_15k)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--target-duels", type=int, default=0,
                    help="Quota de duels pour la passe d'éval (0 = illimité). Affiché en "
                         "progression, bannière 'passe terminée' au quota.")
    ap.add_argument("--html", type=Path,
                    default=Path(__file__).resolve().parents[2] / "arena.html")
    args = ap.parse_args()

    if not (args.campaign / "campaign.json").exists():
        raise SystemExit(f"campaign.json introuvable dans {args.campaign} — "
                         f"lancer d'abord arena.campaign")
    store = VoteStore(args.campaign / "votes.jsonl")
    Handler = make_handler(args.campaign, args.html, store, target_duels=args.target_duels)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[arena] http://127.0.0.1:{args.port}/  campagne={args.campaign} "
          f"({len(store.votes)} votes existants)")
    print("[arena] Ctrl+C pour arrêter")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[arena] arrêté")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
