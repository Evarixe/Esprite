"""Profil torch.profiler enrichi : confirme ou infirme l'hypothèse `.item()` sync.

Mesure deux variantes côte à côte :
  V1 : la boucle de sampling CURRENT (avec .item()) — équivalente au render_samples
  V2 : variante "no-item" où le token reste GPU-side et est .copy_'d dans tok_in

Si V2 << V1 : confirmé, le sync .item() est le goulot.

Output :
  - tableau des kernels par self_cuda_time pour chaque variante
  - tableau des ops par self_cpu_time
  - chrome trace exportée pour chacune
  - estimation du wall-clock per token
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import numpy as np

from genmodel.model import SpriteTransformer
from genmodel.graph_sampler import GraphSampler
from genmodel.vocab import ROLE, PIXELS_PER_FRAME
from genmodel.tokenize import _RASTER_X, _RASTER_Y


device = "cuda"
MAX_LEN = 2500       # cas nominal (prefix ~1050 + 1400 tokens de génération)
PREFIX_LEN = 1050    # taille du préfixe simulé
N_PROF = 100         # iterations profilées
N_WARMUP = 30

print("Loading model bf16 + checkpoint ...")
model = SpriteTransformer().to(device).to(torch.bfloat16).eval()
sd = torch.load("runs/gen_smoke_v2/last.pt", map_location=device, weights_only=False)
model.load_state_dict({k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                       for k, v in sd["model"].items()})

# --- Setup commun : un sampler avec préfixe simulé ---
sampler = GraphSampler(model, max_len=MAX_LEN, dtype=torch.bfloat16)
sampler.reset()

# Préfixe synthétique de PREFIX_LEN tokens type "pixel content"
pre_tokens = [int(np.random.randint(0, 16)) for _ in range(PREFIX_LEN)]
pre_roles  = [ROLE.CONTENT_PIXEL] * PREFIX_LEN
pre_xs     = [int(_RASTER_X[i % PIXELS_PER_FRAME]) for i in range(PREFIX_LEN)]
pre_ys     = [int(_RASTER_Y[i % PIXELS_PER_FRAME]) for i in range(PREFIX_LEN)]
pre_fs     = [0] * PREFIX_LEN
sampler.prepare_from_prefix(pre_tokens, pre_roles, pre_xs, pre_ys, pre_fs)
sampler.capture()
torch.cuda.synchronize()

print(f"Sampler ready: max_len={MAX_LEN}, prefix_len={PREFIX_LEN}, pos_idx={sampler.pos_idx.item()}")


# ============================ V1 : avec .item() (current) ============================

def step_with_item(tok_int):
    """La boucle telle qu'utilisée actuellement : .fill_(int) + replay + .item()."""
    sampler.tok_in.fill_(int(tok_int))
    sampler.x_in.fill_(0)
    sampler.y_in.fill_(0)
    sampler.f_in.fill_(0)
    sampler.role_in.fill_(int(ROLE.CONTENT_PIXEL))
    sampler.graph.replay()
    sampler.pos_idx.add_(1)
    logits = sampler.logits_out[0, 0]
    # Sample + .item() — c'est le sync présumé coupable
    probs = F.softmax(logits.float() / 0.3, dim=-1)
    tok = int(torch.multinomial(probs, num_samples=1).item())
    return tok


# ============================ V2 : sans .item() (proposé) ============================

# Pre-allocate des "constants" GPU pour x/y/f/role qu'on copy_ au lieu de fill_
zero_tensor = torch.zeros(1, 1, dtype=torch.long, device=device)
pixel_role_tensor = torch.full((1, 1), ROLE.CONTENT_PIXEL, dtype=torch.long, device=device)

def step_no_item(tok_tensor):
    """Variante GPU-side : tok_tensor reste sur GPU, pas de .item() dans la boucle."""
    # Copy direct GPU→GPU (D2D)
    sampler.tok_in.copy_(tok_tensor.view(1, 1))
    sampler.x_in.copy_(zero_tensor)
    sampler.y_in.copy_(zero_tensor)
    sampler.f_in.copy_(zero_tensor)
    sampler.role_in.copy_(pixel_role_tensor)
    sampler.graph.replay()
    sampler.pos_idx.add_(1)
    logits = sampler.logits_out[0, 0]
    # Sample sans .item() — token reste GPU tensor
    probs = F.softmax(logits.float() / 0.3, dim=-1)
    tok_tensor_next = torch.multinomial(probs, num_samples=1)   # shape (1,) GPU
    return tok_tensor_next


# ============================ Warmup + bench timing brut ============================

print("\nWarmup V1 + V2...")
for _ in range(N_WARMUP):
    _ = step_with_item(3)
torch.cuda.synchronize()

# Reset pos_idx et mask pour V2 (sinon on continue avec un cache pollué)
sampler.reset()
sampler.prepare_from_prefix(pre_tokens, pre_roles, pre_xs, pre_ys, pre_fs)
# graph est déjà capturé, pas besoin de recapture

tok_t = torch.tensor([3], dtype=torch.long, device=device)
for _ in range(N_WARMUP):
    tok_t = step_no_item(tok_t)
torch.cuda.synchronize()

# --- Wall-clock raw ---
print(f"\n=== Wall-clock ({N_PROF} steps, max_len={MAX_LEN}) ===")

# V1
sampler.reset()
sampler.prepare_from_prefix(pre_tokens, pre_roles, pre_xs, pre_ys, pre_fs)
for _ in range(N_WARMUP):
    _ = step_with_item(3)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_PROF):
    _ = step_with_item(3)
torch.cuda.synchronize()
v1_ms = (time.perf_counter() - t0) / N_PROF * 1000
print(f"  V1 (avec .item()) : {v1_ms:.3f} ms/step")

# V2
sampler.reset()
sampler.prepare_from_prefix(pre_tokens, pre_roles, pre_xs, pre_ys, pre_fs)
tok_t = torch.tensor([3], dtype=torch.long, device=device)
for _ in range(N_WARMUP):
    tok_t = step_no_item(tok_t)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_PROF):
    tok_t = step_no_item(tok_t)
torch.cuda.synchronize()
v2_ms = (time.perf_counter() - t0) / N_PROF * 1000
print(f"  V2 (sans .item()): {v2_ms:.3f} ms/step  ({v1_ms/v2_ms:.2f}x vs V1)")


# ============================ Profil V1 ============================

print("\n--- Profiling V1 (.item) ---")
sampler.reset()
sampler.prepare_from_prefix(pre_tokens, pre_roles, pre_xs, pre_ys, pre_fs)
for _ in range(N_WARMUP): _ = step_with_item(3)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, with_stack=False) as prof_v1:
    for _ in range(N_PROF):
        _ = step_with_item(3)
    torch.cuda.synchronize()

print("\nTop kernels by CUDA self time (V1) :")
print(prof_v1.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
print("\nTop ops by CPU self time (V1) :")
print(prof_v1.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
prof_v1.export_chrome_trace("runs/gen_smoke_v2/trace_v1_item.json")


# ============================ Profil V2 ============================

print("\n--- Profiling V2 (no item) ---")
sampler.reset()
sampler.prepare_from_prefix(pre_tokens, pre_roles, pre_xs, pre_ys, pre_fs)
tok_t = torch.tensor([3], dtype=torch.long, device=device)
for _ in range(N_WARMUP): tok_t = step_no_item(tok_t)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, with_stack=False) as prof_v2:
    for _ in range(N_PROF):
        tok_t = step_no_item(tok_t)
    torch.cuda.synchronize()

print("\nTop kernels by CUDA self time (V2) :")
print(prof_v2.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
print("\nTop ops by CPU self time (V2) :")
print(prof_v2.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
prof_v2.export_chrome_trace("runs/gen_smoke_v2/trace_v2_noitem.json")


print(f"\n=== Summary ===")
print(f"  V1 (with .item per step)     : {v1_ms:.3f} ms/step")
print(f"  V2 (token tensor GPU-side)   : {v2_ms:.3f} ms/step")
print(f"  Speedup V2/V1                : {v1_ms/v2_ms:.2f}x")
print(f"  Chrome traces : runs/gen_smoke_v2/trace_v1_item.json, trace_v2_noitem.json")
