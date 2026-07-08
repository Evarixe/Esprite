"""Canal de contrôle entraînement <-> dashboard.

Le dashboard écrit un fichier `control.json` dans le dossier du run :
  {
    "pause": bool,           # si true, le trainer attend (busy-wait léger)
    "stop": bool,            # si true, le trainer sort proprement après l'epoch courante
    "checkpoint_now": bool,  # si true, le trainer sauve un checkpoint et reset le flag
    "updated_at": float      # timestamp
  }

Le trainer poll ce fichier à chaque fin d'epoch via ControlState.poll().
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time


@dataclass
class ControlState:
    pause: bool = False
    stop: bool = False
    checkpoint_now: bool = False
    updated_at: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "ControlState":
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(**{k: d.get(k, getattr(cls(), k)) for k in cls.__dataclass_fields__})
        except Exception:
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.time()
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def init_control_file(run_dir: Path) -> Path:
    """Écrit un control.json NEUTRE au démarrage — écrase inconditionnellement tout
    flag résiduel (stop/pause/checkpoint_now) laissé par un run ou une session
    précédente dans le même dossier. Sinon un `stop=true` d'avant fait sortir la
    reprise immédiatement (bug observé sur les continuations)."""
    p = run_dir / "control.json"
    ControlState().save(p)
    return p


def poll_and_act(run_dir: Path, pause_log_fn=print, poll_interval: float = 1.0) -> dict:
    """Lit control.json. Si pause=True, bloque jusqu'à pause=False ou stop=True.
    Si checkpoint_now=True, le flag est consommé (remis à False) après lecture.
    Retourne un dict {"stop": bool, "checkpoint_now": bool}.
    """
    p = run_dir / "control.json"
    state = ControlState.load(p)

    if state.pause:
        pause_log_fn("[control] paused — waiting for resume (or stop)")
    while state.pause and not state.stop:
        time.sleep(poll_interval)
        state = ControlState.load(p)

    ckpt = state.checkpoint_now
    if ckpt:
        # Consomme le flag
        state.checkpoint_now = False
        state.save(p)
        pause_log_fn("[control] checkpoint_now consumed")

    return {"stop": state.stop, "checkpoint_now": ckpt}
