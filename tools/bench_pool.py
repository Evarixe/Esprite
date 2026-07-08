"""Test du GraphSamplerPool : promotion dynamique de bucket + timing."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch
import numpy as np
from genmodel.model import SpriteTransformer
from genmodel.graph_sampler import GraphSamplerPool
from genmodel.sample import generate, GenRequest

device = "cuda"
model = SpriteTransformer().to(device).to(torch.bfloat16).eval()
sd = torch.load("runs/gen_smoke_v2/last.pt", map_location=device, weights_only=False)
model.load_state_dict({k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                       for k, v in sd["model"].items()})

data = np.load("data/dataset.npz")
ref = data["cycles"][0, 0]

pool = GraphSamplerPool(model)
print(f"Buckets: {pool.buckets}")

# Le modèle smoke ne s'arrête jamais → génère jusqu'au cap du plus gros bucket.
# On voit donc la promotion complète 0→1→2→3→4.
for run in range(2):
    timings = {}
    req = GenRequest(action="idle", direction="down", frames=2,
                     reference=ref, temperature=0.3, seed=123)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = generate(model, req, device=device, pool=pool, timings=timings)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    n_tok = timings.get("n_tokens_emitted", 0)
    label = "1er run (captures lazy)" if run == 0 else "2e run (graphs cachés)"
    print(f"\n=== {label} ===")
    print(f"  tokens émis     : {n_tok}")
    print(f"  promotions      : {timings.get('promotions', 0)}  (final bucket {timings.get('final_bucket')})")
    print(f"  frames          : {len(result.frames)} ({result.n_valid} valid)")
    print(f"  wall total      : {total:.2f}s")
    print(f"  setup+prefix    : {timings.get('setup_prefix', 0)*1000:.1f} ms")
    print(f"  sample loop     : {timings.get('sample_loop', 0):.3f}s ({timings.get('sample_loop',0)/max(1,n_tok)*1000:.3f} ms/token)")

print(f"\n=== Comparaison flat 18000 (mesuré précédemment) : 27.8s / 1.65 ms/token ===")
