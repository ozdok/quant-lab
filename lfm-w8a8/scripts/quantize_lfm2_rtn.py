#!/usr/bin/env python3
"""quantize_lfm2_rtn.py — Path 1 (W8A8), Path 3 (W8A16-CHANNEL) & Path 4 (W8A8-EXPERTS) RTN for LFM2.5-24B-A2B.

A data-free ``QuantizationModifier(scheme="W8A8", targets=["Linear"])``
with the ignore table from ``lfm2_common``. No calibration corpus, no
SmoothQuant, no GPTQ —
design choice: GPTQ prioritizes by calibration activations and overfits to
corpus, RTN treats every weight equally.

Model-specific decisions, and why:

- **Router ignore pattern.** Other MoE families use ``re:.*mlp[.]gate$``;
  LFM2's router is ``feed_forward.gate``. An ``mlp.gate`` table copied over
  verbatim would silently quantize the router. This uses ``re:.*\\.gate$``.
- **Short-conv block.** LFM2 has ``conv.*`` where other families have names
  like ``linear_attn.*``. The whole block is kept BF16, matching Liquid AI's
  own published FP8 ignore list for this model.
- **No shared expert.** LFM2.5-24B-A2B has 64 routed experts and none
  shared, so there are no ``shared_expert`` keep-outs to carry as dead
  patterns.
- **No processor save.** There is no vision tower; the tokenizer alone is
  what vLLM needs.
- **Concrete model class fails hard.** Falling back to ``AutoModel*`` on
  ImportError demotes ``config.json`` to the text-only config and makes
  vLLM reject the checkpoint. Here the concrete class import is allowed
  to raise.
- **Geometry preflight.** The source index is checked against the expected
  LFM2 geometry before any weights move.

Memory: the 48GB BF16 source does not fit in the build node's RAM, so the model
loads accelerate-split across both GPUs + CPU + disk overflow. Stop any
serving job first so the ~48GB of VRAM is actually free.

``--scheme W8A16-CHANNEL`` builds Path 3 (the weight-only isolation probe /
INT8 W8A16 baseline): Path 1's exact weight quantization (int8, symmetric,
per-channel, static) with the activation quantizer removed, so serving it next
to the FP8 build (which also runs weight-only on Ampere) separates weight
rounding from activation quantization. It is not the stock ``W8A16`` preset,
which is group-128 and would flatter the weights. Its default ``--dst`` is its
own directory and it refuses the shipped W8A8 one.

``--scheme W8A8-EXPERTS`` builds Path 4 (selective component split): all 7296
MoE expert Linears (``w1``, ``w3``, ``w2`` across layers 2-39, ~95.7% of params)
get W8A8 dynamic per-token activations for the INT8 compute fast-path, while the
40 attention Linears (10 layers x q/k/v/out) and 6 dense MLP Linears (layers 0-1
x w1/w3/w2) stay at Path 3's per-channel W8A16. Each vLLM fused unit is uniform
inside itself, avoiding E12's within-expert mixed-scheme wall.

usage: quantize_lfm2_rtn.py [--src PATH] [--dst PATH]
                            [--scheme {W8A8,W8A16-CHANNEL,W8A8-EXPERTS}]
                            [--offload DIR] [--gpu-mem GIB] [--dry-run]
"""

import argparse
import json
import os
import shutil
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfm2_common import (  # noqa: E402
    CENSUS_PATH4_W8A16,
    CENSUS_PATH4_W8A8,
    CENSUS_RTN_TOTAL,
    CONV_LAYER_IDS,
    DEFAULT_SRC,
    DENSE_LAYER_IDS,
    FULL_LAYERS,
    HIDDEN,
    IGNORE_RTN,
    MOE_LAYER_IDS,
    N_ATTN,
    N_EXPERTS,
    N_LAYERS,
    N_MOE_LAYERS,
    PATH4_W8A16,
    PATH4_W8A8,
    ROUTER_SHAPE,
    RTN_SCHEMES,
    assert_not_shipped_dst,
    assert_saved_expert_names,
    assert_saved_scheme,
    default_dst_for,
    layer_prefix,
    require_known_scheme,
    restore_expert_names,
    set_channel_group_size,
)

MODEL_CLASS_IMPORT = "from transformers import Lfm2MoeForCausalLM"


def build_recipe(scheme: str = "W8A8"):
    from llmcompressor.modifiers.quantization import QuantizationModifier

    require_known_scheme(scheme)
    if scheme == "W8A8":
        return QuantizationModifier(
            scheme="W8A8",
            targets=["Linear"],
            ignore=list(IGNORE_RTN),
        )

    from compressed_tensors.quantization import QuantizationScheme
    from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES

    if scheme == "W8A8-EXPERTS":
        group_0 = QuantizationScheme(
            targets=list(PATH4_W8A8),
            weights=PRESET_SCHEMES["W8A8"]["weights"].model_copy(deep=True),
            input_activations=PRESET_SCHEMES["W8A8"][
                "input_activations"].model_copy(deep=True),
        )
        group_1 = QuantizationScheme(
            targets=list(PATH4_W8A16),
            weights=PRESET_SCHEMES["W8A8"]["weights"].model_copy(deep=True),
            input_activations=None,
        )
        return QuantizationModifier(
            config_groups={"group_0": group_0, "group_1": group_1},
            ignore=list(IGNORE_RTN),
        )

    # W8A16-CHANNEL: Path 1's exact weight args, no input quantizer. Built
    # from the W8A8 preset rather than the stock W8A16 one (group-128).
    weight_only = QuantizationScheme(
        targets=["Linear"], weights=PRESET_SCHEMES["W8A8"]["weights"])
    return QuantizationModifier(
        config_groups={"group_0": weight_only},
        ignore=list(IGNORE_RTN),
    )


def _layer_indices(names) -> set:
    """Layer indices present in a safetensors weight map."""
    out = set()
    for n in names:
        if ".layers." not in n:
            continue
        part = n.split(".layers.", 1)[1].split(".", 1)[0]
        if part.isdigit():
            out.add(int(part))
    return out


def preflight_geometry(src: str) -> dict:
    """Assert the source checkpoint really is LFM2.5-24B-A2B.

    AGENTS.md rule: verify geometry against the config at runtime, don't
    trust hardcoded defaults. A wrong-geometry source would quantize
    silently and produce an untrustworthy build, so this fails loud.
    """
    index_path = os.path.join(src, "model.safetensors.index.json")
    with open(index_path) as fh:
        weight_map = json.load(fh)["weight_map"]
    names = set(weight_map)

    layers = _layer_indices(names)
    n_layers = len(layers)

    attn = [i for i in range(N_LAYERS)
            if f"{layer_prefix(i)}.self_attn.q_proj.weight" in names]
    conv = [i for i in range(N_LAYERS)
            if f"{layer_prefix(i)}.conv.in_proj.weight" in names]
    moe = [i for i in range(N_LAYERS)
           if f"{layer_prefix(i)}.feed_forward.experts.0.w1.weight" in names]
    dense = [i for i in range(N_LAYERS)
             if f"{layer_prefix(i)}.feed_forward.w1.weight" in names]

    problems = []
    if n_layers != N_LAYERS:
        problems.append(f"layers {n_layers} != {N_LAYERS}")
    # Compare the actual index sets, not just the counts: a wrong parity
    # (attention on 3,7,...,39) still yields 10 attention and 30
    # convolved layers, so a count-only check passes on the wrong model.
    if attn != FULL_LAYERS:
        problems.append(f"attention layer indices {attn} != {FULL_LAYERS}")
    if conv != CONV_LAYER_IDS:
        problems.append(
            f"short-conv layer indices {conv} != {CONV_LAYER_IDS}")
    if moe != MOE_LAYER_IDS:
        problems.append(f"MoE layer indices {moe} != {MOE_LAYER_IDS}")
    if dense != DENSE_LAYER_IDS:
        problems.append(f"dense layer indices {dense} != {DENSE_LAYER_IDS}")

    # Router shape, expert count, and the absence of foreign-family modules.
    router = f"{layer_prefix(2)}.feed_forward.gate.weight"
    if router not in names:
        problems.append(f"missing router {router}")
    n_experts = len({
        n.split(".experts.")[1].split(".")[0]
        for n in names if ".feed_forward.experts." in n and ".weight" in n
    })
    if n_experts != N_EXPERTS:
        problems.append(f"experts {n_experts} != {N_EXPERTS}")
    for absent in ("shared_expert", "linear_attn", "mtp.", "visual."):
        if any(absent in n for n in names):
            problems.append(f"unexpected {absent!r} tensors (foreign-family source?)")

    if problems:
        raise SystemExit(
            f"geometry preflight FAILED for {src}:\n  - "
            + "\n  - ".join(problems)
        )

    return {
        "layers": n_layers,
        "attn_idx": attn,
        "conv_idx": conv,
        "moe_layers": len(moe),
        "dense_layers": len(dense),
        "experts": n_experts,
        "router_shape": ROUTER_SHAPE,
        "hidden": HIDDEN,
        "tensors": len(names),
    }


def _force_moe_linearize_cpu() -> None:
    """Keep llm-compressor's MoE linearization workspace on CPU.

    oneshot() may fuse the per-expert Linears into 3D expert params and then
    linearize them back out. That stage inherits each module's execution
    device as its offload target — on the build node that piles ~46GB of experts onto
    2x24GB VRAM and OOMs partway through (the datacenter recipe this mirrors
    ran on 96GB+ HBM). Forcing converted experts onto CPU: RTN is per-tensor
    and streams fine from there, and only placement — never values — is
    affected. The saved checkpoint is identical in layout.
    """
    import llmcompressor.modeling.moe.linear_experts as _le

    _orig = _le.get_cache_init_kwargs

    def _cpu_cache(module, default=None):
        kw = dict(_orig(module, default))
        kw["onload_device"] = "cpu"
        kw["offload_device"] = "cpu"
        kw.pop("offload_dir", None)  # DiskCache-only kwarg; CPUCache rejects it
        return kw

    _le.get_cache_init_kwargs = _cpu_cache
    print("moe linearize: converted experts pinned to CPU", flush=True)


def census(model, scheme: str = "W8A8") -> dict:
    """Count quantized Linear modules by class (post-oneshot sanity check)."""
    import torch

    counts = {"experts": 0, "attn": 0, "dense": 0, "conv": 0, "router": 0,
              "other": 0, "fused_experts": 0, "mismatch": []}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        qscheme = getattr(mod, "quantization_scheme", None)
        if qscheme is None:
            continue
        is_a8 = getattr(qscheme, "input_activations", None) is not None
        short = name.replace("model.layers.", "")
        if ".experts." in name:
            if name.endswith(".gate_up_proj"):
                counts["fused_experts"] += 1
            else:
                counts["experts"] += 1
                want_a8 = scheme in ("W8A8", "W8A8-EXPERTS")
                if is_a8 != want_a8:
                    counts["mismatch"].append(short)
        elif ".self_attn." in name:
            counts["attn"] += 1
            want_a8 = (scheme == "W8A8")
            if is_a8 != want_a8:
                counts["mismatch"].append(short)
        elif ".conv." in name:
            counts["conv"] += 1
        elif name.endswith(".gate"):
            counts["router"] += 1
        elif ".feed_forward." in name:
            counts["dense"] += 1
            want_a8 = (scheme == "W8A8")
            if is_a8 != want_a8:
                counts["mismatch"].append(short)
        else:
            counts["other"] += 1
    counts["total"] = (
        counts["experts"] + counts["attn"] + counts["dense"]
        + counts["conv"] + counts["router"] + counts["other"]
    )
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="llm-compressor RTN for LFM2.5-24B-A2B (Path 1 W8A8; "
                    "Path 3 W8A16-CHANNEL probe; Path 4 W8A8-EXPERTS split)")
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=None,
                    help="output dir (default: per scheme; W8A8 -> the "
                         "shipped Path 1 directory)")
    ap.add_argument("--scheme", choices=RTN_SCHEMES, default="W8A8",
                    help="W8A8 = shipped Path 1. W8A16-CHANNEL = Path 3 probe "
                         "with Path 1's weights and no activation quantizer. "
                         "W8A8-EXPERTS = Path 4 experts W8A8 + attn/dense W8A16.")
    ap.add_argument("--offload", default=None,
                    help="accelerate disk-offload dir (default: <dst>.offload)")
    ap.add_argument("--gpu-mem", type=float, default=14,
                    help="GiB per GPU for accelerate max_memory. Keep well below "
                         "24: llm-compressor's MoE linearization stage needs "
                         "several GB of transient CUDA workspace on top.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.dst = args.dst or default_dst_for(args.scheme)
    assert_not_shipped_dst(args.scheme, args.dst)
    recipe = build_recipe(args.scheme)

    if args.dry_run:
        print(f"src:    {args.src}")
        print(f"dst:    {args.dst}")
        print(f"method: QuantizationModifier {args.scheme} RTN "
              "(data-free, no calibration)")
        if args.scheme == "W8A8":
            print("activations: INT8 dynamic per-token")
        elif args.scheme == "W8A8-EXPERTS":
            n_exp = CENSUS_PATH4_W8A8["experts"]
            n_a16 = sum(CENSUS_PATH4_W8A16.values())
            print(f"activations: {n_exp} experts W8A8 + {n_a16} attn/dense "
                  "W8A16 (Path 4 selective split)")
        else:
            print("activations: BF16 (weight-only probe, NOT servable as "
                  "W8A8; same weights as Path 1)")
        print("load:   accelerate auto-split GPU0+GPU1+CPU+disk (48GB src)")
        print(f"model:  {MODEL_CLASS_IMPORT}")
        print("router feed_forward.gate: BF16")
        print("short-conv block (conv.*): BF16  (in_proj/out_proj included)")
        print("leftovers (norms, qk-norms, expert_bias, embed): BF16")
        print(f"expected quantized Linears: {CENSUS_RTN_TOTAL} "
              f"({N_EXPERTS}x{N_MOE_LAYERS} experts x3 + {N_ATTN} attn x4 "
              f"+ {2} dense x3)")
        print("ignore:")
        for pat in IGNORE_RTN:
            print(f"  {pat}")
        print(f"recipe: {recipe}")
        if os.path.isdir(args.src):
            try:
                info = preflight_geometry(args.src)
                print(f"preflight OK: {info}")
            except SystemExit as exc:
                print(f"preflight: {exc}")
        else:
            print(f"preflight: skipped (source not present at {args.src})")
        return

    import torch
    from transformers import AutoTokenizer

    # Concrete class on purpose. AutoModel* would demote config.json to the
    # text-only config and make vLLM reject the checkpoint.
    from transformers import Lfm2MoeForCausalLM as ModelClass

    from llmcompressor import oneshot

    info = preflight_geometry(args.src)
    print(f"geometry preflight OK: {info}", flush=True)

    offload = args.offload or (args.dst + ".offload")
    mm = {i: f"{args.gpu_mem:g}GiB" for i in range(torch.cuda.device_count())}
    mm["cpu"] = "40GiB"
    print(f"loading model (accelerate offload, max_memory={mm})...", flush=True)
    model = ModelClass.from_pretrained(
        args.src,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=mm,
        offload_folder=offload,
        offload_state_dict=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)

    _force_moe_linearize_cpu()

    oneshot(
        model=model,
        recipe=recipe,
        tokenizer=tokenizer,
        output_dir=args.dst,
        trust_remote_code_model=True,
    )

    counts = census(model, scheme=args.scheme)
    print(f"\n=== post-oneshot census ===\n  {counts}", flush=True)
    exp = {"experts": N_EXPERTS * N_MOE_LAYERS * 3,
           "attn": N_ATTN * 4,
           "dense": 2 * 3,
           "conv": 0,
           "router": 0}
    bad = {k: (counts[k], v) for k, v in exp.items() if counts[k] != v}
    if counts["fused_experts"]:
        bad["fused_experts"] = (counts["fused_experts"], 0)
    if counts["mismatch"]:
        bad["mismatch"] = (len(counts["mismatch"]), counts["mismatch"][:5])
    if bad:
        print(f"WARNING: census mismatch {bad} — do NOT trust this build.",
              flush=True)
    else:
        print("census OK (experts/attn/dense quantized; conv/router kept BF16)",
              flush=True)

    # Linearization rename-back: oneshot saves gate_proj/up_proj/down_proj;
    # transformers + vLLM expect w1/w3/w2. Without this the 7296 experts
    # do not load (2026-10-14 ship-blocker).
    n_fixed = restore_expert_names(args.dst)
    print(f"expert names restored: {n_fixed} tensors -> w1/w3/w2", flush=True)
    assert_saved_expert_names(args.dst)
    if args.scheme in ("W8A16-CHANNEL", "W8A8-EXPERTS"):
        changed = set_channel_group_size(args.dst)
        print("channel-wise group_size -> -1 for vLLM 0.29.0: "
              f"{'written' if changed else 'already set'}", flush=True)
    assert_saved_scheme(args.dst, args.scheme)
    print(f"saved scheme verified: {args.scheme}", flush=True)
    # Preserve source generation artifacts: oneshot regenerates
    # generation_config.json (temp drifted 0.1 -> 0.6 on the shipped build).
    for _f in ("generation_config.json", "chat_template.jinja",
               "tokenizer.json", "tokenizer_config.json"):
        _s, _d = os.path.join(args.src, _f), os.path.join(args.dst, _f)
        if os.path.isfile(_s) and not os.path.isfile(_d):
            shutil.copy2(_s, _d)
            print(f"preserved {_f} from source", flush=True)

    # NOTE: deliberately no in-process generate here. Forwarding the
    # just-compressed model fires compressed-tensors' decompress hook (full
    # INT8->BF16 in memory) and OOM-killed a 64GiB host at 82%.
    # Validate in a fresh process (scripts/smoke.py or vLLM).
    print(f"done -> {args.dst}")


if __name__ == "__main__":
    main()
