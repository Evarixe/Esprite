"""Profil rapide du sampling : où passe le temps par token ?

On instrumente UNE génération courte (frames=1, ~1024 tokens à générer après le préfixe)
et on mesure :
  A. Temps total
  B. Temps moyen par token
  C. Décomposition par step : tensor alloc, model forward, sampling, accounting

Total budget : ~15s.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from genmodel.model import SpriteTransformer
from genmodel.sample import build_prefix, GenRequest, _FORBIDDEN_IN_GEN, _sample_token
from genmodel.vocab import (
    PIXELS_PER_FRAME, ROLE, MAX_FRAMES, is_pixel_token,
    FRAME_SEP, SEQ_END,
)
from genmodel.tokenize import _RASTER_X, _RASTER_Y


def profile_one_cycle(model, ref, device, n_tokens_target=1024, seed=42):
    """Génère n_tokens_target tokens en mesurant le breakdown par step.
    Retourne dict avec timings cumulés par phase."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    req = GenRequest(action="idle", direction="down", frames=1,
                     reference=ref, temperature=0.3, seed=seed)
    tokens, roles, xs, ys, fs, _ = build_prefix(req)
    prefix_len = len(tokens)

    # --- Forward initial ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_tens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    x_tens = torch.tensor(xs,     dtype=torch.long, device=device).unsqueeze(0)
    y_tens = torch.tensor(ys,     dtype=torch.long, device=device).unsqueeze(0)
    f_tens = torch.tensor(fs,     dtype=torch.long, device=device).unsqueeze(0)
    r_tens = torch.tensor(roles,  dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, kv_cache = model(t_tens, x_tens, y_tens, f_tens, r_tens,
                                  attn_mask=None, return_cache=True)
    torch.cuda.synchronize()
    t_prefix = time.perf_counter() - t0
    last_logits = logits[0, -1].float()

    timings = {"alloc": 0.0, "forward": 0.0, "sample": 0.0, "post": 0.0, "n_steps": 0}
    current_frame_idx = 0
    pixel_in_current = 0
    cur_seq_pos = prefix_len

    t_loop_start = time.perf_counter()
    for n_new in range(n_tokens_target):
        # --- A. Sample previous logits ---
        torch.cuda.synchronize()
        a0 = time.perf_counter()
        tok = _sample_token(last_logits, req.temperature, gen, forbid_ids=_FORBIDDEN_IN_GEN)
        torch.cuda.synchronize()
        timings["sample"] += time.perf_counter() - a0

        # --- B. Accounting + position computation ---
        b0 = time.perf_counter()
        if is_pixel_token(tok):
            role_new = ROLE.CONTENT_PIXEL
            x_new = int(_RASTER_X[pixel_in_current % PIXELS_PER_FRAME])
            y_new = int(_RASTER_Y[pixel_in_current % PIXELS_PER_FRAME])
            f_new = min(current_frame_idx, MAX_FRAMES - 1)
            pixel_in_current += 1
        else:
            role_new = ROLE.CONTENT_SEP
            x_new = 0; y_new = 0
            f_new = min(current_frame_idx, MAX_FRAMES - 1)
            if tok == FRAME_SEP:
                current_frame_idx += 1
                pixel_in_current = 0
            elif tok == SEQ_END:
                break
        timings["post"] += time.perf_counter() - b0

        # --- C. Alloc input tensors ---
        c0 = time.perf_counter()
        t_in = torch.tensor([[tok]],      dtype=torch.long, device=device)
        x_in = torch.tensor([[x_new]],    dtype=torch.long, device=device)
        y_in = torch.tensor([[y_new]],    dtype=torch.long, device=device)
        f_in = torch.tensor([[f_new]],    dtype=torch.long, device=device)
        r_in = torch.tensor([[role_new]], dtype=torch.long, device=device)
        torch.cuda.synchronize()
        timings["alloc"] += time.perf_counter() - c0

        # --- D. Forward 1 token avec cache ---
        d0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, kv_cache = model(t_in, x_in, y_in, f_in, r_in,
                                      attn_mask=None, kv_cache=kv_cache,
                                      seq_offset=cur_seq_pos)
        torch.cuda.synchronize()
        timings["forward"] += time.perf_counter() - d0
        last_logits = logits[0, -1].float()
        cur_seq_pos += 1
        timings["n_steps"] += 1

    t_loop = time.perf_counter() - t_loop_start
    return {
        "prefix_forward_s": t_prefix,
        "loop_total_s": t_loop,
        "n_steps": timings["n_steps"],
        "alloc_s": timings["alloc"],
        "forward_s": timings["forward"],
        "sample_s": timings["sample"],
        "post_s": timings["post"],
    }


def main():
    device = "cuda"
    model = SpriteTransformer().to(device).eval()
    sd = torch.load("runs/gen_smoke_v2/last.pt", map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    data = np.load("data/dataset.npz")
    ref = data["cycles"][0, 0]

    # Warmup (le 1er run inclut compile + init)
    _ = profile_one_cycle(model, ref, device, n_tokens_target=64, seed=1)

    # Mesure réelle sur 1024 tokens
    r = profile_one_cycle(model, ref, device, n_tokens_target=1024, seed=42)
    n = r["n_steps"]
    print(f"=== Profil 1 cycle, {n} tokens générés ===")
    print(f"Prefix forward      : {r['prefix_forward_s']*1000:8.1f} ms")
    print(f"Loop total          : {r['loop_total_s']*1000:8.1f} ms  ({r['loop_total_s']/n*1000:.2f} ms/token)")
    print(f"  forward (cached)  : {r['forward_s']*1000:8.1f} ms  ({r['forward_s']/n*1000:.2f} ms/token)  {r['forward_s']/r['loop_total_s']*100:5.1f}%")
    print(f"  tensor alloc      : {r['alloc_s']*1000:8.1f} ms  ({r['alloc_s']/n*1000:.2f} ms/token)  {r['alloc_s']/r['loop_total_s']*100:5.1f}%")
    print(f"  sampling          : {r['sample_s']*1000:8.1f} ms  ({r['sample_s']/n*1000:.2f} ms/token)  {r['sample_s']/r['loop_total_s']*100:5.1f}%")
    print(f"  accounting        : {r['post_s']*1000:8.1f} ms  ({r['post_s']/n*1000:.2f} ms/token)  {r['post_s']/r['loop_total_s']*100:5.1f}%")
    measured = r["forward_s"] + r["alloc_s"] + r["sample_s"] + r["post_s"]
    print(f"  sum mesuré        : {measured*1000:8.1f} ms  ({(r['loop_total_s']-measured)*1000:.1f} ms non-attribué)")


if __name__ == "__main__":
    main()
