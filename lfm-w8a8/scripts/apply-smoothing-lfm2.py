#!/usr/bin/env python3
"""apply-smoothing-lfm2.py — SmoothQuant folds for LFM2.5-24B-A2B (Path 2).

Not required for Path 1 (RTN). Ran end to end 2026-10-16/17; that SQ-W8A8
build regressed (KLD 0.70 vs 0.22) because fold A left the router unscaled.
Read the post-mortem in docs/accuracy-suggestions.md before rebuilding.

Every fold is mathematically exact pre-quantization; tests/test_fold_plan.py
pins which tensors each fold touches and that the block outputs do not move.
Scale convention: s = a_max^ALPHA / w_max^(1-ALPHA), capped to
[S_MIN, S_MAX], s=1 where a_max == 0; w_max over the QUANTIZED consumers'
columns only.

  A moe-in   : ``ffn_norm`` (PLAIN RMSNorm: w' = w / s) -> expert w1 cols * s,
               expert w3 cols * s AND the router ``feed_forward.gate`` cols *
               s. One shared s_a per layer, because the same normed tensor
               feeds the router and every expert: any consumer left out sees
               x / s (measured: top-4 expert set changed for 62% of tokens).
  C moe-down : per-expert s_c[e]. Divide ONLY ``w3`` (up) rows by s_c[e] and
               multiply ``w2`` cols by s_c[e]. The down input is
               silu(gate) * up, so scaling the up side passes straight
               through the product; scaling the gate side would NOT (SiLU is
               nonlinear). See docs/accuracy-suggestions.md item 1 for the
               fused-halves fold this avoids — structurally impossible here
               because LFM2 ships gate and up as separate tensors.
  B attn-in  : ``operator_norm`` (PLAIN) -> q/k/v proj cols * s. All three
               are required. q/k pass through q_layernorm / k_layernorm, but
               s is per-channel rather than a scalar, so RMSNorm's scale
               invariance does not cancel it: with q/k left unfolded they see
               x / s (21% attention-block error; folded as coded: 4e-7).
  E attn-out : v rows / t -> o cols * t, GQA-constrained: t is [N_KV, HD]
               built from the group max so every q-head sharing a v channel
               shares one scale. v_proj is [N_KV*HD, HIDDEN] so
               t.reshape(-1) aligns with its rows.

NOT touched: the entire short-conv block, ``expert_bias``, embeddings, q/k
layernorms, and the two dense MLPs (0-1). The router is folded by A but stays
BF16 and unquantized. Output is stored BF16 (training dtype — better range
for folded gains).

usage: apply-smoothing-lfm2.py [--src PATH] [--dst PATH] [--scales PATH]
                               [--folds ABCE] [--dry-run]
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfm2_common import (  # noqa: E402
    DEFAULT_SRC,
    G,
    HD,
    HIDDEN,
    MOE_INTER,
    N_EXPERTS,
    N_KV,
    N_LAYERS,
    layer_prefix,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

OUT_DTYPE = torch.bfloat16


def plan_ops(folds: str, weight_map: dict, scales: dict) -> dict:
    """Plan every requested fold as {tensor name: [(kind, vec), ...]}.

    Pure bookkeeping: no weights are read, so tests can pin which tensors each
    fold touches. Folds are independent of each other; steps for one tensor
    are applied in order by ``apply_ops``.
    """
    ops: dict = {}

    def add(name, kind, vec):
        ops.setdefault(name, []).append((kind, vec))

    moe_layers = [i for i in range(N_LAYERS)
                  if f"{layer_prefix(i)}.feed_forward.experts.0.w1.weight"
                  in weight_map]
    attn_layers = [i for i in range(N_LAYERS)
                   if f"{layer_prefix(i)}.self_attn.q_proj.weight"
                   in weight_map]
    assert len(moe_layers) == 38, len(moe_layers)
    assert len(attn_layers) == 10, len(attn_layers)

    for i in moe_layers:
        lk = layer_prefix(i)
        s_a = scales[f"{lk}.feed_forward.experts"]
        s_c = scales[f"{lk}.feed_forward.experts.w2"]
        assert tuple(s_a.shape) == (HIDDEN,), tuple(s_a.shape)
        assert tuple(s_c.shape) == (N_EXPERTS, MOE_INTER), tuple(s_c.shape)

        # --- A: fold into the norm, then push s_a into EVERY consumer of the
        # normed tensor: the router as well as both expert inputs ---
        if "A" in folds:
            add(f"{lk}.ffn_norm.weight", "norm_plain", s_a)
            add(f"{lk}.feed_forward.gate.weight", "colmul", s_a)
            for e in range(N_EXPERTS):
                add(f"{lk}.feed_forward.experts.{e}.w1.weight", "colmul", s_a)
                add(f"{lk}.feed_forward.experts.{e}.w3.weight", "colmul", s_a)
        # --- C: up rows out, down cols in (exactly cancels) ---
        if "C" in folds:
            for e in range(N_EXPERTS):
                add(f"{lk}.feed_forward.experts.{e}.w3.weight",
                    "rowdiv", s_c[e])
                add(f"{lk}.feed_forward.experts.{e}.w2.weight",
                    "colmul", s_c[e])

    for i in attn_layers:
        lk = layer_prefix(i)
        s_b = scales[f"{lk}.self_attn.q_proj"]
        t = scales[f"{lk}.self_attn.out_proj"]
        assert tuple(t.shape) == (N_KV, HD), tuple(t.shape)

        if "B" in folds:
            add(f"{lk}.operator_norm.weight", "norm_plain", s_b)
            for p in ("q_proj", "k_proj", "v_proj"):
                add(f"{lk}.self_attn.{p}.weight", "colmul", s_b)
        # E: v rows out, o cols in.
        if "E" in folds:
            add(f"{lk}.self_attn.v_proj.weight", "rowdiv", t.reshape(-1))
            add(f"{lk}.self_attn.out_proj.weight", "colmul",
                t.unsqueeze(1).expand(N_KV, G, HD).reshape(-1))
    return ops


def apply_ops(x: torch.Tensor, steps: list) -> torch.Tensor:
    """Apply one tensor's planned (kind, vec) steps, in order."""
    for kind, vec in steps:
        v = vec.float() if isinstance(vec, torch.Tensor) else vec
        if kind == "norm_plain":
            x = x / v
        elif kind == "colmul":
            x = x * v.unsqueeze(0)
        elif kind == "rowdiv":
            x = x / v.unsqueeze(1)
        else:  # pragma: no cover - guarded by construction
            raise AssertionError(f"unknown op {kind}")
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=os.path.join(
        os.path.dirname(DEFAULT_SRC), "LFM2.5-24B-A2B-smoothed"))
    ap.add_argument("--scales", default=None)
    ap.add_argument("--folds", default="ABCE",
                    help="subset of folds to apply (e.g. E, B, A, C)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.folds = "".join(sorted(set(args.folds.upper()) - set(" ")))
    assert set(args.folds) <= set("ABCE") and args.folds, args.folds

    SRC, DST = Path(args.src), Path(args.dst)
    if set(args.folds) != set("ABCE"):
        DST = Path(str(DST) + "-" + args.folds)
        print(f"subset folds {args.folds} -> {DST}", flush=True)
    if args.scales is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.scales = os.path.normpath(
            os.path.join(here, "..", "calib", "scales-lfm2.pt"))

    if args.dry_run:
        print(f"src:    {SRC}")
        print(f"dst:    {DST}")
        print(f"scales: {args.scales}")
        print(f"geometry: G={G} HD={HD} N_KV={N_KV}, out dtype {OUT_DTYPE}")
        print("folds:  A moe-in (shared s_a: ffn_norm + router gate + expert "
              "w1/w3), C moe-down (per-expert, w3 rows ONLY), B attn-in "
              "(q/k/v cols), E attn-out (GQA)")
        print("untouched: short-conv block, expert_bias, embeddings, q/k "
              "layernorms, dense MLPs 0-1")
        print("norms: PLAIN RMSNorm (w' = w/s) — LFM2 is NOT zero-centered")
        print("--dry-run: exiting before model/scales load")
        return

    wm = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]

    t0 = time.time()
    DST.mkdir(parents=True, exist_ok=True)
    CH = {k: (v.float() if isinstance(v, torch.Tensor) else v)
          for k, v in torch.load(args.scales).items()}

    ops = plan_ops(args.folds, wm, CH)
    print(f"plan: {len(ops)} tensors ({time.time()-t0:.0f}s)", flush=True)

    by_shard: dict = {}
    for tname, shard in wm.items():
        by_shard.setdefault(shard, []).append(tname)

    applied = 0
    for shard, names in sorted(by_shard.items()):
        tensors = load_file(str(SRC / shard))
        changed = 0
        for n in names:
            if n not in ops:
                continue
            tensors[n] = apply_ops(tensors[n].float(), ops[n]).to(OUT_DTYPE)
            changed += 1
        applied += changed
        save_file(tensors, str(DST / shard))
        print(f"  {shard}: {changed} transformed", flush=True)
        del tensors

    assert applied == len(ops), f"applied {applied} != planned {len(ops)}"

    for f in SRC.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, DST / f.name)
    cfg = json.load(open(DST / "config.json"))
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(DST / "config.json", "w"), indent=2)
    print(f"TOTAL {time.time()-t0:.0f}s -> {DST} (bf16, {applied} tensors)",
          flush=True)


if __name__ == "__main__":
    main()
