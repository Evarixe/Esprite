"""Warmup GPU progressif après changement de driver.

Joue tous les chemins critiques (forward+backward training, KV cache graph capture,
matmul Q=1) sur les shapes exactes qu'on utilisera en production, pour :
  - primer cuDNN/cuBLAS kernel selection (autotune sur shape réelle)
  - valider que la nouvelle stack driver + torch nightly cu128 est stable
  - sortir tout de suite en cas d'incompatibilité, pas au milieu d'un training

Aucun checkpoint écrit, aucun gradient appliqué. Pur smoke test ~1-2 min.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from genmodel.model import SpriteTransformer
from genmodel.graph_sampler import GraphSampler
from genmodel.vocab import ROLE


def banner(s): print(f"\n=== {s} ===")


def step1_context_init():
    banner("1. CUDA context init")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: pas de CUDA dispo, abandon")
        sys.exit(1)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    # small matmul pour init le contexte sans risquer un gros kernel
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    t0 = time.perf_counter()
    y = x @ x
    torch.cuda.synchronize()
    print(f"small matmul (64x64 bf16): {(time.perf_counter()-t0)*1000:.2f} ms — OK")


def step2_model_init():
    banner("2. Model construction + bf16 cast")
    t0 = time.perf_counter()
    model = SpriteTransformer().to("cuda").to(torch.bfloat16).eval()
    torch.cuda.synchronize()
    print(f"model init + bf16: {time.perf_counter()-t0:.2f} s, params {model.n_params/1e6:.2f}M")
    return model


def step3_load_checkpoint(model):
    banner("3. Reload checkpoint gen_200b/last.pt")
    ckpt = Path("runs/gen_200b/last.pt")
    if not ckpt.exists():
        print(f"WARN: checkpoint absent, skip (chemin: {ckpt})")
        return
    t0 = time.perf_counter()
    state = torch.load(ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict({k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                           for k, v in state["model"].items()})
    torch.cuda.synchronize()
    print(f"checkpoint loaded: {time.perf_counter()-t0:.2f} s, step {state.get('step', '?')}")


def step4_train_forward_backward(model):
    banner("4. 5× forward+backward training-shape (batch 16, seq ~3081)")
    # Le checkpoint est en eval bf16 — pour train il faut grads + adam.
    # On utilise un modèle SÉPARÉ en fp32 pour éviter de toucher le model audit.
    train_model = SpriteTransformer().cuda()
    opt = torch.optim.AdamW(train_model.parameters(), lr=1e-4)

    B, T = 16, 3081
    tokens = torch.randint(0, 36, (B, T), device="cuda")
    xp = torch.randint(0, 32, (B, T), device="cuda")
    yp = torch.randint(0, 32, (B, T), device="cuda")
    fp = torch.zeros(B, T, dtype=torch.long, device="cuda")
    roles = torch.full((B, T), ROLE.CONTENT_PIXEL, dtype=torch.long, device="cuda")
    attn = torch.ones(B, T, dtype=torch.bool, device="cuda")

    times = []
    for i in range(5):
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = train_model(tokens, xp, yp, fp, roles, attn)
            loss = logits.float().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"per-step (5 steps): {[f'{t:.0f}ms' for t in times]}  (1er = chaud cuDNN)")
    print(f"steady-state ~{sum(times[1:])/4:.0f} ms/step")
    # libère VRAM
    del train_model, opt, logits, loss
    torch.cuda.empty_cache()


def step5_graph_capture(model):
    banner("5. CUDA Graph capture + replay (sampling)")
    sampler = GraphSampler(model, max_len=3500)
    pre_tokens = [3]*1031
    pre_roles  = [ROLE.CONTENT_PIXEL]*1031
    pre_x = [i % 32 for i in range(1031)]
    pre_y = [(i // 32) % 32 for i in range(1031)]
    pre_f = [0]*1031
    t0 = time.perf_counter()
    sampler.prepare_from_prefix(pre_tokens, pre_roles, pre_x, pre_y, pre_f)
    torch.cuda.synchronize()
    print(f"prefix forward + buffers fill: {(time.perf_counter()-t0)*1000:.1f} ms")
    t0 = time.perf_counter()
    sampler.capture()
    torch.cuda.synchronize()
    print(f"graph capture (3 warmup + 1 capture): {(time.perf_counter()-t0)*1000:.1f} ms")
    # 100 step replays
    t0 = time.perf_counter()
    for _ in range(100):
        _ = sampler.step(3, 0, 0, 0, ROLE.CONTENT_PIXEL)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 100 * 1000
    print(f"100 step replays: {dt:.3f} ms/token (attendu ~1.0-1.2 ms à max_len=3500)")
    del sampler
    torch.cuda.empty_cache()


def main():
    step1_context_init()
    model = step2_model_init()
    step3_load_checkpoint(model)
    step4_train_forward_backward(model)
    step5_graph_capture(model)
    banner("DONE — tous les chemins critiques OK")
    print(f"VRAM résiduelle : {torch.cuda.memory_allocated()/1e9:.2f} GB allouée, "
          f"{torch.cuda.memory_reserved()/1e9:.2f} GB réservée")


if __name__ == "__main__":
    main()
