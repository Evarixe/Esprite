"""HeartbeatWriter — écrit un petit JSON d'avancement live, throttlé à ~4 Hz.

Écriture atomique via os.replace pour éviter qu'un read concurrent du serveur
voie un fichier tronqué.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any


class HeartbeatWriter:
    def __init__(self, path: Path, min_interval_s: float = 0.25, ema_alpha: float = 0.1):
        self.path = path
        self.min_interval_s = min_interval_s
        self.alpha = ema_alpha
        self._state: dict[str, Any] = {}
        self._last_write_ts = 0.0
        self._last_step_ts: float | None = None
        path.parent.mkdir(parents=True, exist_ok=True)

    def _ema(self, key: str, value: float) -> float:
        # Ignore les NaN (ex: pendant les pauses) pour ne pas empoisonner l'EMA.
        if value != value:  # NaN
            return self._state.get(key, 0.0)
        prev = self._state.get(key)
        if prev is None:
            self._state[key] = float(value)
        else:
            self._state[key] = (1 - self.alpha) * float(prev) + self.alpha * float(value)
        return self._state[key]

    def update(self, *, step: int, total_steps: int, bucket: str, loss: float,
               top1: float, n_tokens: int, dt: float, lr: float, gpu: dict | None,
               force: bool = False) -> None:
        # EMAs sur stats running (NaN ignorés via _ema)
        self._ema("loss_running", loss)
        self._ema("top1_running", top1)
        if n_tokens > 0:   # pendant une pause, on ne touche pas aux débits
            self._ema("step_per_sec_ema", 1.0 / max(1e-6, dt))
            self._ema("tokens_per_sec_ema", n_tokens / max(1e-6, dt))
        self._state.setdefault("step_per_sec_ema", 0.0)
        self._state.setdefault("tokens_per_sec_ema", 0.0)

        # Throttle
        now = time.time()
        if not force and (now - self._last_write_ts) < self.min_interval_s:
            return

        payload = {
            "ts": now,
            "step": int(step),
            "total_steps": int(total_steps),
            "bucket_last": bucket,
            "loss_running": round(self._state["loss_running"], 4),
            "top1_running": round(self._state["top1_running"], 4),
            "step_per_sec_ema": round(self._state["step_per_sec_ema"], 3),
            "tokens_per_sec_ema": int(self._state["tokens_per_sec_ema"]),
            "lr": float(lr),
            "gpu": gpu,  # dict or None
        }

        # Écriture atomique. Sur Windows, os.replace échoue (PermissionError /
        # WinError 5) si un lecteur tient le fichier cible ouvert à cet instant
        # (le serveur monitor ou un poller externe). On retente brièvement ; en
        # dernier recours on skippe ce heartbeat (purement informatif, jamais
        # bloquant pour le training).
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(tmp, self.path)
                break
            except PermissionError:
                if attempt == 4:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return  # skip ce heartbeat, on réessaiera au prochain step
                time.sleep(0.02)
        self._last_write_ts = now
