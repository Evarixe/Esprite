"""Wrapper minimal NVML pour stats GPU.

Lazy init au premier appel. Renvoie None proprement si pas de GPU NVIDIA
ou si pynvml manque (training CPU possible sans crasher).
"""
from __future__ import annotations
from typing import Optional

_pynvml = None
_handle = None
_failed = False


def _init() -> bool:
    global _pynvml, _handle, _failed
    if _failed:
        return False
    if _handle is not None:
        return True
    try:
        import pynvml
        pynvml.nvmlInit()
        _pynvml = pynvml
        _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return True
    except Exception:
        _failed = True
        return False


def gpu_stats() -> Optional[dict]:
    """Retourne {util_pct, mem_used_gb, mem_total_gb, temp_c, power_w, power_limit_w} ou None."""
    if not _init():
        return None
    try:
        util = _pynvml.nvmlDeviceGetUtilizationRates(_handle).gpu
        mem = _pynvml.nvmlDeviceGetMemoryInfo(_handle)
        temp = _pynvml.nvmlDeviceGetTemperature(_handle, _pynvml.NVML_TEMPERATURE_GPU)
        GIB = 1024 ** 3
        out = {
            "util_pct": int(util),
            "mem_used_gb": round(mem.used / GIB, 2),   # GiB pour matcher nvidia-smi
            "mem_total_gb": round(mem.total / GIB, 2),
            "temp_c": int(temp),
        }
        # Puissance (W) : draw instantané + cap actif. mW → W. Best-effort (NVML
        # peut ne pas l'exposer sur certains GPU/drivers) → on n'échoue pas pour ça.
        try:
            out["power_w"] = round(_pynvml.nvmlDeviceGetPowerUsage(_handle) / 1000.0)
            out["power_limit_w"] = round(_pynvml.nvmlDeviceGetEnforcedPowerLimit(_handle) / 1000.0)
        except Exception:
            pass
        return out
    except Exception:
        return None
