#!/usr/bin/env python3
"""capture-lfm2.py — per-hook activation capture for LFM2.5-24B-A2B (Path 2).

Path 2 step 1. Not required for Path 1 (RTN).
Ran end to end 2026-10-16/17; see docs/accuracy-suggestions.md (build log and
post-mortem) before reusing its output.

Runs calibration windows through the BF16 model and records, per hook, the
per-window channel maxima plus a sample of activation rows.

Hooks:

  {lk}.feed_forward                 routed-expert input = post-``ffn_norm``
                                    output, shared by all 64 experts (A fold)
  (C fold: no hook; lfm2_alpha.py recomputes it from the rows above)
  {lk}.self_attn.q_proj             attn input (B fold, attn layers only)
  {lk}.self_attn.out_proj           attn output (E fold, attn layers only)

No per-expert ``w2`` hooks: LFM2 ships experts as 2D Linears on disk, but
transformers fuses them into 3D Parameters on load, so no per-expert module
exists at runtime. Broadcasting the 2048-wide routed-in stats onto the
1536-wide down inputs is dimensionally invalid, so lfm2_alpha.py recomputes
every expert's down input in exact BF16 from these shared rows and the stored
w1/w3.

Memory: capture loads with the same accelerate split as the quantizers
(per-GPU caps + CPU + disk overflow), because the 48GB source does not fit in
the build node's RAM. ``--sample-experts`` bounds the sample-row shards; per-window
maxima are kept for every expert.

usage: capture-lfm2.py [--src PATH] [--corpus PATH] [--out-dir DIR]
                       [--seq-len 2048] [--n-windows 96]
                       [--rows-per-window 128] [--sample-experts 8]
                       [--gpu-mem 14] [--dry-run]
"""

import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfm2_common import (  # noqa: E402
    DEFAULT_SRC,
    N_ATTN,
    N_CONV,
    N_EXPERTS,
    N_LAYERS,
    N_MOE_LAYERS,
    layer_prefix,
)

MODEL_CLASS_IMPORT = "from transformers import Lfm2MoeForCausalLM"
DEFAULT_CORPUS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "calib", "calib_mixed.txt"))


def hook_names_for(prefix: str, is_attn: bool, is_moe: bool,
                   sample_experts: int) -> tuple:
    """Return (winmax_hooks, sample_hooks).

    ``winmax_hooks`` get per-window amax for every window. ``sample_hooks``
    additionally get sample rows (bounded by ``sample_experts``).

    Layers 0-1 have a dense MLP and no expert modules, so they contribute
    only the routed-input hook (which for them is the dense MLP input).
    """

    # Dense MLPs (layers 0-1) are RTN-only: no smoothing folds touch them,
    # so capturing their inputs wastes windows. MoE layers capture the
    # routed-expert input; attn layers add their two hooks.
    base = [f"{prefix}.feed_forward"] if is_moe else []
    if is_attn:
        base += [f"{prefix}.self_attn.q_proj", f"{prefix}.self_attn.out_proj"]
    # C has no hook: per-expert w2 modules do not exist after HF load (fused
    # 3D params), so only the shared routed-in hook is captured. Alpha
    # recomputes every expert's C input from these shared rows.
    return base, list(base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-windows", type=int, default=96)
    ap.add_argument("--rows-per-window", type=int, default=128)
    ap.add_argument("--sample-experts", type=int, default=8,
                    help="experts per layer whose sample rows are kept")
    ap.add_argument("--shard-windows", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=14,
                    help="GiB per GPU for accelerate max_memory (an uncapped "
                         "capture OOMs on a 48GB source)")
    ap.add_argument("--offload", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.out_dir is None:
        args.out_dir = os.path.normpath(os.path.join(here, "..", "calib", "acts"))

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        if args.dry_run:
            print(f"src:     {args.src} (absent — geometry check skipped)")
            print(f"corpus:  {args.corpus}")
            print(f"out-dir: {args.out_dir}")
            print(f"hooks:   1 routed-in + {N_EXPERTS} expert-down "
                  f"+ 2 attn per attn layer")
            print("--dry-run: exiting")
            return
        raise SystemExit(f"no safetensors index at {index_path}")

    wm = json.load(open(index_path))["weight_map"]
    layers = [layer_prefix(i) for i in range(N_LAYERS)]
    kinds = ["attn" if f"{lk}.self_attn.q_proj.weight" in wm else "conv"
             for lk in layers]
    assert kinds.count("attn") == N_ATTN and kinds.count("conv") == N_CONV, (
        f"attn/conv {kinds.count('attn')}/{kinds.count('conv')}, "
        f"expected {N_ATTN}/{N_CONV}")

    winmax_hooks, sample_hooks = [], []
    n_moe_seen = 0
    for lk, k in zip(layers, kinds):
        is_moe = f"{lk}.feed_forward.experts.0.w1.weight" in wm
        n_moe_seen += int(is_moe)
        w, s = hook_names_for(lk, k == "attn", is_moe, args.sample_experts)
        winmax_hooks += w
        sample_hooks += s
    assert n_moe_seen == N_MOE_LAYERS, (n_moe_seen, N_MOE_LAYERS)
    assert len(winmax_hooks) == len(set(winmax_hooks)), "duplicate hooks"

    if args.dry_run:
        print(f"src:     {args.src}")
        print(f"corpus:  {args.corpus}")
        print(f"out-dir: {args.out_dir}")
        print(f"layers:  {kinds.count('attn')} attn + {kinds.count('conv')} conv; "
              f"{n_moe_seen} MoE + {N_LAYERS - n_moe_seen} dense")
        print(f"winmax hooks: {len(winmax_hooks)} "
              f"(38 MoE routed-in + 10 attn x 2; C recomputed in alpha)")
        print("--dry-run: exiting before model load")
        return
    offload = args.offload or (args.out_dir + ".offload")
    mm = {i: f"{args.gpu_mem:g}GiB" for i in range(torch.cuda.device_count())}
    mm["cpu"] = "40GiB"
    print(f"loading model (accelerate split, max_memory={mm})...", flush=True)
    try:
        from transformers import Lfm2MoeForCausalLM as ModelClass
    except ImportError:
        try:
            from transformers import AutoModelForMultimodalLM as ModelClass
        except ImportError:
            from transformers import AutoModelForCausalLM as ModelClass
    model = ModelClass.from_pretrained(
        args.src, dtype=torch.bfloat16, device_map="auto",
        max_memory=mm, offload_folder=offload, offload_state_dict=True,
        trust_remote_code=True)
    model.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    text = open(args.corpus, errors="replace").read()
    ids = tok(text, add_special_tokens=False).input_ids
    span = len(ids) - args.seq_len
    assert span > 0, f"corpus {args.corpus} yields no {args.seq_len}-token window"
    step = max(1, span // args.n_windows)
    wins = [ids[s:s + args.seq_len] for s in range(0, span, step)][:args.n_windows]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"capture: {len(wins)} x {args.seq_len} windows "
          f"(strided, step {step})", flush=True)

    modules = dict(model.named_modules())
    missing = [h for h in winmax_hooks if h not in modules]
    assert not missing, f"hook modules missing: {missing[:5]}"

    cur: dict = {}
    winmax: dict = {h: [] for h in winmax_hooks}
    rows: dict = {h: [] for h in sample_hooks}
    shard_idx = 0

    def make_hook(h):
        def fn(mod, inp, out):
            x = inp[0].detach()
            if isinstance(x, tuple):
                x = x[0]
            cur[h] = x.reshape(-1, x.shape[-1]).float().cpu()
        return fn

    for h in winmax_hooks:
        modules[h].register_forward_hook(make_hook(h))

    def amax_or_none(h, width):
        x = cur.get(h)
        if x is None or x.numel() == 0:
            return torch.zeros(width)
        return x.amax(0)

    with torch.no_grad():
        for wi, w in enumerate(wins):
            cur.clear()
            inp = torch.tensor([w], device=model.device)
            model(input_ids=inp)
            for h in winmax_hooks:
                x = cur.get(h)
                if x is None or x.numel() == 0:
                    winmax[h].append(torch.zeros(0))
                    continue
                winmax[h].append(x.amax(0))
                if h in rows:
                    n = min(args.rows_per_window, x.shape[0])
                    sel = torch.linspace(0, x.shape[0] - 1, n).long()
                    rows[h].append(x[sel])
            if (wi + 1) % args.shard_windows == 0 or wi + 1 == len(wins):
                torch.save({h: torch.cat(rows[h]) for h in sample_hooks
                            if rows[h]},
                           os.path.join(args.out_dir,
                                        f"actsample-{shard_idx}.pt"))
                rows = {h: [] for h in sample_hooks}
                shard_idx += 1
            if (wi + 1) % 16 == 0:
                print(f"  {wi + 1}/{len(wins)} windows", flush=True)

    torch.save({h: torch.stack(winmax[h]) for h in winmax_hooks if winmax[h]},
               os.path.join(args.out_dir, "actmax-winmax.pt"))
    print(f"saved winmax + {shard_idx} sample shards -> {args.out_dir}",
          flush=True)
    print("NOTE: C inputs are recomputed per expert in lfm2_alpha.py from "
          "the shared routed-in rows.", flush=True)


if __name__ == "__main__":
    main()
