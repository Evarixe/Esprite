"""Test de parité et benchmark du GraphSampler.

3 paths à comparer sur un même préfixe + suite de N tokens :
  A. dynamic cache (kv_cache concat, baseline actuelle)
  B. static cache eager (GraphSampler sans .capture(), pour isoler le gain alloc)
  C. static cache + CUDA Graphs

Mesure : ms/token, gain vs A, parité numérique (max abs diff dans logits).
"""
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
N_STEPS = 200          # tokens à générer pour mesure
SEED = 42


def random_pixel_step(step):
    """Génère des inputs pseudo-réalistes pour 1 step (pixel content)."""
    return {
        "tok":  3,                                # token pixel arbitraire
        "x":    int(_RASTER_X[step % PIXELS_PER_FRAME]),
        "y":    int(_RASTER_Y[step % PIXELS_PER_FRAME]),
        "f":    0,
        "role": ROLE.CONTENT_PIXEL,
    }


def path_A_dynamic(model, prefix, N):
    """Baseline : kv_cache dynamique avec concat."""
    t, x, y, f, r = prefix
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, kv_cache = model(t, x, y, f, r, attn_mask=None, return_cache=True)
    torch.cuda.synchronize()

    seq_off = t.shape[1]
    last_logits = None
    t0 = time.perf_counter()
    for step in range(N):
        s = random_pixel_step(step)
        t_in = torch.tensor([[s["tok"]]],  dtype=torch.long, device=device)
        x_in = torch.tensor([[s["x"]]],    dtype=torch.long, device=device)
        y_in = torch.tensor([[s["y"]]],    dtype=torch.long, device=device)
        f_in = torch.tensor([[s["f"]]],    dtype=torch.long, device=device)
        r_in = torch.tensor([[s["role"]]], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, kv_cache = model(t_in, x_in, y_in, f_in, r_in,
                                      attn_mask=None, kv_cache=kv_cache,
                                      seq_offset=seq_off)
        seq_off += 1
        last_logits = logits[0, -1].float().clone()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1000, last_logits


def path_static(model, prefix, N, use_graph: bool):
    """static cache eager (use_graph=False) ou CUDA Graphs (True)."""
    t_list, x_list, y_list, f_list, r_list = prefix
    sampler = GraphSampler(model, max_len=t_list.shape[1] + N + 8, dtype=torch.bfloat16)
    sampler.reset()
    prefix_tokens = t_list[0].cpu().tolist()
    prefix_roles  = r_list[0].cpu().tolist()
    prefix_x      = x_list[0].cpu().tolist()
    prefix_y      = y_list[0].cpu().tolist()
    prefix_f      = f_list[0].cpu().tolist()
    L = sampler.prepare_from_prefix(prefix_tokens, prefix_roles, prefix_x, prefix_y, prefix_f)

    if use_graph:
        sampler.capture()
        # Après capture, le warmup a écrit dans les buffers à position L,
        # mais pos_idx est toujours = L (le warmup n'incrémente pas).
        # Donc on est en bon état pour step.

    torch.cuda.synchronize()
    last_logits = None
    t0 = time.perf_counter()
    for step in range(N):
        s = random_pixel_step(step)
        logits = sampler.step(s["tok"], s["x"], s["y"], s["f"], s["role"])
        last_logits = logits.float().clone()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1000, last_logits


def main():
    torch.manual_seed(SEED)
    print("Loading model + checkpoint ...")
    model = SpriteTransformer().to(device).to(torch.bfloat16).eval()
    sd = torch.load("runs/gen_smoke_v2/last.pt", map_location=device, weights_only=False)
    # Le checkpoint est fp32, on charge dans le modèle bf16 — torch convertit
    model.load_state_dict({k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                           for k, v in sd["model"].items()})

    # Prefix : ~1050 tokens (idle/down + ref pour réalisme)
    data = np.load("data/dataset.npz")
    from genmodel.sample import build_prefix, GenRequest
    req = GenRequest(action="idle", direction="down", frames=4,
                     reference=data["cycles"][0, 0], temperature=0.3, seed=SEED)
    tokens, roles, xs, ys, fs, _ = build_prefix(req)
    L = len(tokens)
    print(f"Prefix length: {L}")

    t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    x = torch.tensor(xs,     dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ys,     dtype=torch.long, device=device).unsqueeze(0)
    f = torch.tensor(fs,     dtype=torch.long, device=device).unsqueeze(0)
    r = torch.tensor(roles,  dtype=torch.long, device=device).unsqueeze(0)
    prefix = (t, x, y, f, r)

    # Warmup chacun
    print("\nWarming up paths...")
    _ = path_A_dynamic(model, prefix, 20)
    _ = path_static(model, prefix, 20, use_graph=False)

    print(f"\n=== Benchmark {N_STEPS} steps ===")
    ms_A, last_A = path_A_dynamic(model, prefix, N_STEPS)
    print(f"  A. dynamic kv_cache    : {ms_A:.3f} ms/token")
    ms_B, last_B = path_static(model, prefix, N_STEPS, use_graph=False)
    print(f"  B. static cache eager  : {ms_B:.3f} ms/token  ({ms_A/ms_B:.2f}x vs A)")

    try:
        ms_C, last_C = path_static(model, prefix, N_STEPS, use_graph=True)
        print(f"  C. static + CUDA graph : {ms_C:.3f} ms/token  ({ms_A/ms_C:.2f}x vs A)")
        diff_BA = (last_A - last_B).abs().max().item()
        diff_CA = (last_A - last_C).abs().max().item()
    except Exception as e:
        print(f"  C. graph capture failed: {type(e).__name__}: {e}")
        diff_BA = (last_A - last_B).abs().max().item()
        diff_CA = None

    print("\n=== Parité numérique (last token logits) ===")
    print(f"  A vs B (static eager): max abs diff = {diff_BA:.4e}")
    if diff_CA is not None:
        print(f"  A vs C (graph)       : max abs diff = {diff_CA:.4e}")


if __name__ == "__main__":
    main()
