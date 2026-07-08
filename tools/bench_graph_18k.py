"""Bench à max_len=18000 — le cas catastrophique précédent."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch
import numpy as np
from genmodel.model import SpriteTransformer
from genmodel.graph_sampler import GraphSampler
from genmodel.vocab import ROLE, PIXELS_PER_FRAME
from genmodel.tokenize import _RASTER_X, _RASTER_Y


device = "cuda"
MAX_LEN = 18000
PREFIX_LEN = 1031
N_PROF = 200

model = SpriteTransformer().to(device).to(torch.bfloat16).eval()
sd = torch.load("runs/gen_smoke_v2/last.pt", map_location=device, weights_only=False)
model.load_state_dict({k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                       for k, v in sd["model"].items()})

sampler = GraphSampler(model, max_len=MAX_LEN, dtype=torch.bfloat16)
sampler.reset()
pre_tokens = [int(np.random.randint(0, 16)) for _ in range(PREFIX_LEN)]
pre_roles  = [ROLE.CONTENT_PIXEL] * PREFIX_LEN
pre_xs     = [int(_RASTER_X[i % PIXELS_PER_FRAME]) for i in range(PREFIX_LEN)]
pre_ys     = [int(_RASTER_Y[i % PIXELS_PER_FRAME]) for i in range(PREFIX_LEN)]
pre_fs     = [0] * PREFIX_LEN
sampler.prepare_from_prefix(pre_tokens, pre_roles, pre_xs, pre_ys, pre_fs)
sampler.capture()
torch.cuda.synchronize()

# warmup
for _ in range(20):
    _ = sampler.step(3, 0, 0, 0, ROLE.CONTENT_PIXEL)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N_PROF):
    _ = sampler.step(3, 0, 0, 0, ROLE.CONTENT_PIXEL)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N_PROF * 1000
print(f"max_len={MAX_LEN}: {dt:.3f} ms/token  ({N_PROF} steps)")
print(f"  Projection : 1 cycle de 2050 tokens = {dt*2050/1000:.2f}s")
print(f"  Projection : 1 cycle de 16600 tokens = {dt*16600/1000:.2f}s")
