#!/usr/bin/env python3
"""lfm2_alpha.py — pick per-hook (scale-base, alpha) OFFLINE for LFM2 (Path 2).

Not required for Path 1 (RTN). Ran 2026-10-16/17.

KNOWN DEFECT: this grid has no identity candidate and builds its activation
stats from true window maxima. On held-out rows the scales it chose were
WORSE than plain RTN for fold A (26 of 38 layers), C and E; only B helped.
See the post-mortem in docs/accuracy-suggestions.md before regenerating
scales.

For each hook and each candidate s(base, alpha), simulate the fold + INT8
quantization on captured rows, with no build/eval cycle per candidate:

    X' = X / s        (per-token dynamic INT8, symmetric, amax/127)
    W' = W * s        (per-out-channel INT8, symmetric)
    err = ||q(X') @ q(W').T - X @ W.T||_F^2 / ||X @ W.T||_F^2

Candidates: base in {absmax, median-of-window-maxima} x alpha in
{0.5, 0.65, 0.8, 0.9}. (absmax, 0.8) is the reference cell.

Classes:
  moe-in   (A) hook {lk}.feed_forward — shared by all 64 experts, so wmax
               pools over every expert's w1 + w3 columns.
  moe-down (C) per-expert down inputs RECOMPUTED in exact BF16 from the
               shared routed-in rows (silu(X @ w1.T) * (X @ w3.T)); no w2
               hooks exist once HF fuses the experts. The (base, alpha) pair
               is chosen on expert 0, then each expert gets its own scale
               from its own amax over ALL sampled rows (routing only selects
               outputs, so that is a conservative source).
  attn-in  (B) hook {lk}.self_attn.q_proj; consumers q/k/v stacked.
  attn-out (E) hook {lk}.self_attn.out_proj, GQA-constrained to
               [N_KV=8, G=4, HD=64].

Output: calib/scales-lfm2.pt {smoothing-hook: chosen s}

usage: lfm2_alpha.py [--src PATH] [--acts-dir DIR] [--out PATH] [--dry-run]
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfm2_common import (  # noqa: E402
    ALPHAS,
    DEFAULT_SRC,
    G,
    HD,
    MOE_INTER,
    N_ATTN,
    N_EXPERTS,
    N_KV,
    N_LAYERS,
    N_Q,
    S_MAX,
    S_MIN,
    layer_prefix,
    pop_hook_samples,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

DEV = "cuda:0"
# Cap on proxy rows. The moe-in job concatenates 64 experts' consumer rows;
# full sample depth would blow 24GB. The proxy metric stays representative.
PROXY_ROWS = 4096


def stride_rows(X):
    """Strided subsample to PROXY_ROWS (head-take lets one domain dominate)."""
    if X.shape[0] <= PROXY_ROWS:
        return X
    import torch
    sel = torch.linspace(0, X.shape[0] - 1, PROXY_ROWS).long()
    return X[sel]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--acts-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.acts_dir is None:
        args.acts_dir = os.path.normpath(os.path.join(here, "..", "calib", "acts"))
    if args.out is None:
        args.out = os.path.normpath(
            os.path.join(here, "..", "calib", "scales-lfm2.pt"))

    if args.dry_run:
        print(f"src:      {args.src}")
        print(f"acts-dir: {args.acts_dir}")
        print(f"out:      {args.out}")
        print(f"geometry: {N_LAYERS} layers ({N_ATTN} attn + 30 conv), "
              f"{N_EXPERTS} experts, N_KV={N_KV} G={G} HD={HD}")
        print(f"grid:     base in {{absmax, median}} x alpha in {ALPHAS}")
        print("classes:  moe-in (A), moe-down per-expert (C), attn-in (B), "
              "attn-out (E)")
        print("no GDN class: LFM2's 30 non-attention layers are short convs "
              "and stay BF16")
        print("--dry-run: exiting before data load")
        return

    t0 = time.time()
    winmax = torch.load(os.path.join(args.acts_dir, "actmax-winmax.pt"))
    samples: dict = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.acts_dir, "actsample-*.pt"))):
        for k, v in torch.load(f).items():
            samples[k].append(v)
    print(f"loaded {len(winmax)} winmax hooks ({time.time()-t0:.0f}s)", flush=True)

    idx = json.load(open(os.path.join(args.src,
                                      "model.safetensors.index.json")))["weight_map"]

    def read_t(name):
        with safe_open(os.path.join(args.src, idx[name]), framework="pt") as f:
            return f.get_tensor(name).float()

    def q_rows(t):
        s = t.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
        return (t / s).round().clamp(-127, 127) * s

    def err_for(X, W, s):
        Xf, Wf = X / s, W * s
        ref = X @ W.T
        got = q_rows(Xf) @ q_rows(Wf).T
        return ((got - ref).pow(2).sum() / ref.pow(2).sum()).item()

    def calc_s(a, wmax, alpha):
        s = a.pow(alpha) / wmax.clamp(min=1e-8).pow(1 - alpha)
        return torch.where(a > 0, s.clamp(min=S_MIN, max=S_MAX),
                           torch.ones_like(s))

    def best_cell(X, W, wm, wmax, constraint=None):
        best = (None, None, float("inf"), None, None)
        ref_err = None
        for base_name, a_full in (("absmax", wm.amax(0)),
                                  ("median", wm.median(0).values)):
            for alpha in ALPHAS:
                if constraint == "gqa":
                    a_c = a_full.view(N_KV, G, HD).amax(1)
                    w_c = wmax.view(N_KV, G, HD).amax(1)
                    s_c = calc_s(a_c, w_c, alpha)
                    s = s_c.unsqueeze(1).expand(N_KV, G, HD).reshape(-1)
                else:
                    s_c = s = calc_s(a_full, wmax, alpha)
                e = err_for(X, W, s)
                if base_name == "absmax" and alpha == 0.8:
                    ref_err = e
                if e < best[2]:
                    best = (base_name, alpha, e, s_c.cpu(), s.cpu())
        return best, ref_err

    chosen: dict = {}
    stats = defaultdict(lambda: defaultdict(int))
    gains = defaultdict(list)

    attn_layers = [i for i in range(N_LAYERS)
                   if f"{layer_prefix(i)}.self_attn.q_proj.weight" in idx]
    moe_layers = [i for i in range(N_LAYERS)
                  if f"{layer_prefix(i)}.feed_forward.experts.0.w1.weight" in idx]

    for li, i in enumerate(range(N_LAYERS)):
        lk = layer_prefix(i)
        is_attn = i in attn_layers
        is_moe = i in moe_layers

        # ---- A: routed-expert input (shared by all experts) ----
        # The two dense layers (0-1) carry only ~0.6% of the parameters and
        # have no expert table, so they are left unfolded and quantized by
        # plain RTN. Smoothing them would need a separate dense-down hook
        # for under 1% of the model.
        if is_moe:
            hook = f"{lk}.feed_forward"
            X = stride_rows(pop_hook_samples(samples, hook).to(DEV))
            wm = winmax[hook].to(DEV)
            gu = torch.cat([
                read_t(f"{lk}.feed_forward.experts.{e}.{p}.weight").to(DEV)
                for e in range(N_EXPERTS) for p in ("w1", "w3")
            ], 0)
            wmax = gu.abs().amax(0)
            best, ref_err = best_cell(X, gu, wm, wmax)
            chosen[f"{lk}.feed_forward.experts"] = best[3]
            stats["moe-in"][f"{best[0]}/a{best[1]}"] += 1
            gains["moe-in"].append((ref_err - best[2]) / max(ref_err, 1e-12))
            Xc_keep = X  # C reuses these routed-in rows; freed below
            del gu, wm
            torch.cuda.empty_cache()

            # ---- C: per-expert moe-down ----
            # Pick (base, alpha) on the sampled experts; then give every
            # expert its own scale from its own recorded amax.
            # C has no hooks (experts are fused on load): every expert's
            # down input is recomputed below from the shared routed-in rows.
            # Each expert keeps its own weight-max; folds stay exact.
            hook_c = f"{lk}.feed_forward"
            s_c = torch.ones(N_EXPERTS, MOE_INTER)
            if hook_c in winmax:
                # C inputs recomputed, not hooked: down_in[e] =
                # silu(X @ w1[e].T) * (X @ w3[e].T) in exact BF16 on the
                # shared routed-in rows. Routing only selects outputs, so
                # full-distribution amaxes are a conservative scale source.
                Xe0 = None
                amaxs = []
                with torch.no_grad():
                    for e in range(N_EXPERTS):
                        W1 = read_t(f"{lk}.feed_forward.experts.{e}.w1.weight"
                                    ).to(DEV)
                        W3 = read_t(f"{lk}.feed_forward.experts.{e}.w3.weight"
                                    ).to(DEV)
                        XX = Xc_keep.to(DEV)
                        d = torch.nn.functional.silu(XX @ W1.T) * (XX @ W3.T)
                        amaxs.append(d.abs().amax(0).cpu())
                        if e == 0:
                            Xe0 = d[:4096].cpu()
                        del W1, W3, XX, d
                torch.cuda.empty_cache()
                wm_stack = torch.stack(amaxs)
                del amaxs
                W0 = read_t(f"{lk}.feed_forward.experts.0.w2.weight").to(DEV)
                best_c, ref_err_c = best_cell(
                    Xe0.to(DEV), W0, wm_stack.amax(0).to(DEV),
                    W0.abs().amax(0))
                del W0, Xe0
                torch.cuda.empty_cache()
                base_name, alpha = best_c[0], best_c[1]
                a_med = wm_stack.median(0).values
                for e in range(N_EXPERTS):
                    a_full = (wm_stack[e] if base_name == "absmax" else a_med)
                    wmax_e = read_t(f"{lk}.feed_forward.experts.{e}.w2.weight"
                                    ).abs().amax(0)
                    s_c[e] = calc_s(a_full, wmax_e, alpha)
                del a_med, wm_stack, Xc_keep
                torch.cuda.empty_cache()
                stats["moe-down"][f"{base_name}/a{alpha}"] += 1
                gains["moe-down"].append(
                    (ref_err_c - best_c[2]) / max(ref_err_c, 1e-12))
            chosen[f"{lk}.feed_forward.experts.w2"] = s_c

        # ---- B / E: attention ----
        if is_attn:
            hook_b = f"{lk}.self_attn.q_proj"
            Xb = stride_rows(pop_hook_samples(samples, hook_b).to(DEV))
            Wb = torch.cat([read_t(f"{lk}.self_attn.{p}.weight").to(DEV)
                            for p in ("q_proj", "k_proj", "v_proj")], 0)
            best_b, ref_b = best_cell(Xb, Wb, winmax[hook_b].to(DEV),
                                      Wb.abs().amax(0))
            chosen[f"{lk}.self_attn.q_proj"] = best_b[3]
            stats["attn-in"][f"{best_b[0]}/a{best_b[1]}"] += 1
            gains["attn-in"].append((ref_b - best_b[2]) / max(ref_b, 1e-12))
            del Wb, Xb
            torch.cuda.empty_cache()

            hook_e = f"{lk}.self_attn.out_proj"
            Xe = stride_rows(pop_hook_samples(samples, hook_e).to(DEV))
            Wo = read_t(f"{lk}.self_attn.out_proj.weight").to(DEV)
            assert Wo.shape[1] == N_Q * HD, (Wo.shape, N_Q * HD)
            best_e, ref_e = best_cell(Xe, Wo, winmax[hook_e].to(DEV),
                                      Wo.abs().amax(0), constraint="gqa")
            t = best_e[3]
            assert tuple(t.shape) == (N_KV, HD), tuple(t.shape)
            chosen[f"{lk}.self_attn.out_proj"] = t
            stats["attn-out"][f"{best_e[0]}/a{best_e[1]}"] += 1
            gains["attn-out"].append((ref_e - best_e[2]) / max(ref_e, 1e-12))
            del Wo, Xe
            torch.cuda.empty_cache()

        if (li + 1) % 8 == 0:
            print(f"  {li+1}/{N_LAYERS} layers ({time.time()-t0:.0f}s)",
                  flush=True)

    torch.save(chosen, args.out)
    print("\nchosen cells per class:", flush=True)
    for cls, d in stats.items():
        g = torch.tensor(gains[cls])
        print(f"  {cls:<10} {dict(d)}  proxy-err reduction vs (absmax,0.8): "
              f"mean {100*g.mean():.1f}% max {100*g.max():.1f}%", flush=True)
    print(f"saved {args.out} ({len(chosen)} hooks). TOTAL {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
