"""Profil kernel-level rapide du forward 1-token avec cache.

Utilise torch.profiler pour identifier les ops dominantes en wall-clock.
Output : top 15 kernels par self_cuda_time.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.profiler import profile, ProfilerActivity
from genmodel.model import SpriteTransformer
from genmodel.vocab import ROLE


device = "cuda"
prefix_len = 1050
print("Loading model (bf16 static) ...")
m = SpriteTransformer().to(device).to(torch.bfloat16).eval()

# Inputs
t = torch.randint(0, 16, (1, prefix_len), device=device)
x = torch.zeros(1, prefix_len, dtype=torch.long, device=device)
y = torch.zeros(1, prefix_len, dtype=torch.long, device=device)
f = torch.zeros(1, prefix_len, dtype=torch.long, device=device)
r = torch.full((1, prefix_len), ROLE.CONTENT_PIXEL, dtype=torch.long, device=device)

with torch.no_grad():
    _, cache = m(t, x, y, f, r, attn_mask=None, return_cache=True)
torch.cuda.synchronize()

one_t = torch.zeros(1, 1, dtype=torch.long, device=device)
one_x = torch.zeros(1, 1, dtype=torch.long, device=device)
one_y = torch.zeros(1, 1, dtype=torch.long, device=device)
one_f = torch.zeros(1, 1, dtype=torch.long, device=device)
one_r = torch.full((1, 1), ROLE.CONTENT_PIXEL, dtype=torch.long, device=device)

seq_off = prefix_len
# warmup
with torch.no_grad():
    for _ in range(10):
        _, cache = m(one_t, one_x, one_y, one_f, one_r,
                      attn_mask=None, kv_cache=cache, seq_offset=seq_off)
        seq_off += 1
torch.cuda.synchronize()

print("Profiling 50 iters ...")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False) as prof:
    with torch.no_grad():
        for _ in range(50):
            _, cache = m(one_t, one_x, one_y, one_f, one_r,
                          attn_mask=None, kv_cache=cache, seq_offset=seq_off)
            seq_off += 1
    torch.cuda.synchronize()

print("\n=== Top kernels by CUDA self time ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
