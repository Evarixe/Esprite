"""Types partagés pour le pipeline dataset."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class RawCycle:
    """Cycle brut sorti d'un parser de source.

    frames : liste d'arrays (H, W, 4) RGBA uint8. Toutes frames d'un cycle ont la
    même taille. La transparence est portée par le canal alpha (alpha=0 = transparent).
    """
    frames: list[np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def native_size(self) -> int:
        return self.frames[0].shape[0]

    @property
    def n_frames(self) -> int:
        return len(self.frames)


@dataclass
class ProcessedCycle:
    """Cycle après quantification de palette et downsampling vers 32x32.

    indices : array (n_frames, 32, 32) uint8, valeurs 0..15 (0 = transparent).
    palette_rgb : array (16, 3) uint8, palette RGB indexée. Index 0 = (0,0,0) par convention.
    """
    indices: np.ndarray
    palette_rgb: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
