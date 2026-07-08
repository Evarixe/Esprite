"""Mesure d'UNE génération complète qui tape le cap (~16400 tokens) à max_len=18000.

Instrumente via le dict `timings` de generate() + mesure wall-clock totale.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch
import numpy as np
from genmodel.model import SpriteTransformer
from genmodel.graph_sampler import GraphSampler
from genmodel.sample import generate, GenRequest

device = "cuda"
model = SpriteTransformer().to(device).to(torch.bfloat16).eval()
sd = torch.load("runs/gen_smoke_v2/last.pt", map_location=device, weights_only=False)
model.load_state_dict({k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                       for k, v in sd["model"].items()})

data = np.load("data/dataset.npz")
ref = data["cycles"][0, 0]

# Sampler réutilisable, max_len=18000
sampler = GraphSampler(model, max_len=18000, dtype=torch.bfloat16)

req = GenRequest(action="idle", direction="down", frames=2,
                 reference=ref, temperature=0.3, seed=123)

# Première génération = inclut capture du graph + 1ère résidence (à part)
timings = {}
torch.cuda.synchronize()
t0 = time.perf_counter()
result = generate(model, req, device=device, sampler=sampler, timings=timings)
torch.cuda.synchronize()
total = time.perf_counter() - t0

n_tok = timings.get("n_tokens_emitted", 0)
print(f"=== Génération unique, max_len=18000 ===")
print(f"  tokens émis        : {n_tok}")
print(f"  frames produites   : {len(result.frames)} ({result.n_valid} valid / {result.n_invalid} invalid)")
print(f"  stop_reason        : {result.stop_reason}")
print(f"  wall-clock total   : {total:.2f}s")
print(f"  --- breakdown ---")
print(f"  capture graph (1x) : {timings.get('capture', 0)*1000:8.1f} ms")
print(f"  setup+prefix load  : {timings.get('setup_prefix', 0)*1000:8.1f} ms")
print(f"  prefix forward     : {timings.get('prefix_forward', 0)*1000:8.1f} ms")
print(f"  sample loop        : {timings.get('sample_loop', 0):8.3f} s   ({timings.get('sample_loop',0)/max(1,n_tok)*1000:.3f} ms/token)")

# 2e génération : sampler déjà capturé, mesure le régime "chaud"
timings2 = {}
torch.cuda.synchronize()
t0 = time.perf_counter()
result2 = generate(model, req, device=device, sampler=sampler, timings=timings2)
torch.cuda.synchronize()
total2 = time.perf_counter() - t0
n_tok2 = timings2.get("n_tokens_emitted", 0)
print(f"\n=== 2e génération (graph déjà capturé) ===")
print(f"  tokens émis        : {n_tok2}")
print(f"  wall-clock total   : {total2:.2f}s")
print(f"  capture            : {timings2.get('capture', 0)*1000:.1f} ms (0 attendu, déjà capturé)")
print(f"  sample loop        : {timings2.get('sample_loop', 0):.3f}s   ({timings2.get('sample_loop',0)/max(1,n_tok2)*1000:.3f} ms/token)")
