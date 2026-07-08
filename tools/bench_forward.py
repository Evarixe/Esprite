"""Benchmark ciblé du forward 1-token avec cache, sur plusieurs configs :
  A. Baseline : modèle fp32, autocast bf16 dans le call
  B. Modèle cast à bf16 statique, pas d'autocast
  C. B + torch.compile (mode 'reduce-overhead' pour favoriser le launch overhead)
  D. B + torch.compile (mode 'max-autotune')

Mesure : ms/token sur 200 iterations cached forward, après warmup.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from genmodel.model import SpriteTransformer
from genmodel.vocab import ROLE


def build_dummy_inputs(device, prefix_len, n_iters):
    """Renvoie (prefix_tensors, single_token_tensors)."""
    t = torch.randint(0, 16, (1, prefix_len), device=device)
    x = torch.zeros(1, prefix_len, dtype=torch.long, device=device)
    y = torch.zeros(1, prefix_len, dtype=torch.long, device=device)
    f = torch.zeros(1, prefix_len, dtype=torch.long, device=device)
    r = torch.full((1, prefix_len), ROLE.CONTENT_PIXEL, dtype=torch.long, device=device)

    one_t = torch.zeros(1, 1, dtype=torch.long, device=device)
    one_x = torch.zeros(1, 1, dtype=torch.long, device=device)
    one_y = torch.zeros(1, 1, dtype=torch.long, device=device)
    one_f = torch.zeros(1, 1, dtype=torch.long, device=device)
    one_r = torch.full((1, 1), ROLE.CONTENT_PIXEL, dtype=torch.long, device=device)
    return (t, x, y, f, r), (one_t, one_x, one_y, one_f, one_r)


def bench(model, prefix, one, n_iters, autocast=True, label=""):
    device = "cuda"
    t_full, x_full, y_full, f_full, r_full = prefix
    one_t, one_x, one_y, one_f, one_r = one

    # Forward initial
    if autocast:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, cache = model(t_full, x_full, y_full, f_full, r_full,
                              attn_mask=None, return_cache=True)
    else:
        with torch.no_grad():
            _, cache = model(t_full, x_full, y_full, f_full, r_full,
                              attn_mask=None, return_cache=True)
    torch.cuda.synchronize()

    # warmup loop
    seq_off = t_full.shape[1]
    for _ in range(20):
        if autocast:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, cache = model(one_t, one_x, one_y, one_f, one_r,
                                  attn_mask=None, kv_cache=cache, seq_offset=seq_off)
        else:
            with torch.no_grad():
                _, cache = model(one_t, one_x, one_y, one_f, one_r,
                                  attn_mask=None, kv_cache=cache, seq_offset=seq_off)
        seq_off += 1
    torch.cuda.synchronize()

    # mesure
    t0 = time.perf_counter()
    for _ in range(n_iters):
        if autocast:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, cache = model(one_t, one_x, one_y, one_f, one_r,
                                  attn_mask=None, kv_cache=cache, seq_offset=seq_off)
        else:
            with torch.no_grad():
                _, cache = model(one_t, one_x, one_y, one_f, one_r,
                                  attn_mask=None, kv_cache=cache, seq_offset=seq_off)
        seq_off += 1
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iters * 1000
    print(f"  {label:40s}: {dt:.3f} ms/token")
    return dt


def main():
    device = "cuda"
    prefix_len = 1050
    n_iters = 200

    # ----------- A. fp32 model + autocast bf16 -----------
    print("Loading model (fp32) ...")
    m_a = SpriteTransformer().to(device).eval()
    sd = torch.load("runs/gen_smoke_v2/last.pt", map_location=device, weights_only=False)
    m_a.load_state_dict(sd["model"])
    prefix, one = build_dummy_inputs(device, prefix_len, n_iters)
    print("\n== Bench ==")
    t_a = bench(m_a, prefix, one, n_iters, autocast=True, label="A. fp32 + autocast bf16")

    # ----------- B. bf16 static -----------
    m_b = m_a.to(torch.bfloat16)
    t_b = bench(m_b, prefix, one, n_iters, autocast=False, label="B. bf16 static, no autocast")

    # ----------- C. bf16 + torch.compile(reduce-overhead) -----------
    try:
        m_c = torch.compile(m_b, mode="reduce-overhead", fullgraph=False)
        t_c = bench(m_c, prefix, one, n_iters, autocast=False, label="C. bf16 + compile(reduce-overhead)")
    except Exception as e:
        print(f"  C. compile failed: {type(e).__name__}: {e}")
        t_c = None

    # ----------- D. bf16 + torch.compile(max-autotune) -----------
    try:
        m_d = torch.compile(m_b, mode="max-autotune", fullgraph=False)
        t_d = bench(m_d, prefix, one, n_iters, autocast=False, label="D. bf16 + compile(max-autotune)")
    except Exception as e:
        print(f"  D. compile failed: {type(e).__name__}: {e}")
        t_d = None

    print(f"\nbaseline A : {t_a:.3f} ms/token")
    if t_b is not None: print(f"  B speedup : {t_a/t_b:.2f}x")
    if t_c is not None: print(f"  C speedup : {t_a/t_c:.2f}x")
    if t_d is not None: print(f"  D speedup : {t_a/t_d:.2f}x")


if __name__ == "__main__":
    main()
