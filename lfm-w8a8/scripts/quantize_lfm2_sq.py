#!/usr/bin/env python3
"""quantize_lfm2_sq.py — Path 2: RTN with E12 mixed-precision groups (LFM2).

Ran 2026-10-16/17: builds and passes the census (4908 W8A8 / 2434 W8A16) but
vLLM 0.29.0 cannot serve it — fused MoE needs ONE scheme across gate/up/down.
For a servable smoothed build run quantize_lfm2_rtn.py on the smoothed
checkpoint instead. Path 1 (quantize_lfm2_rtn.py) is the daily driver.

Takes the smoothed checkpoint from apply-smoothing-lfm2.py and applies
uniform RTN INT8 — NO GPTQ (design choice: GPTQ prioritizes by calibration
activations and overfits to the corpus; RTN treats every weight equally).

Groups:
  group_0 W8A8  (INT8 weights + INT8 dynamic per-token activations):
                attention q/k/v/o, expert w1/w3 (gate/up), dense w1/w3.
  group_1 W8A16 (INT8 weights, BF16 activations): expert w2 and dense w2
                (the down projections — gated-product inputs, spiky,
                position-dynamic error; the Freaksterz E12 finding).

Router, the short-conv block, embeddings and norms stay BF16 via IGNORE_SQ.

The census asserts the full split (experts AND attention AND dense), not just
the experts — a silent mis-quant of the attention half must not print
"census OK" (docs/accuracy-suggestions.md item 4).

Memory: accelerate-split load, same as the RTN applier.

usage: quantize_lfm2_sq.py [--model PATH] [--out PATH] [--offload DIR]
                           [--gpu-mem GIB] [--dry-run]
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from compressed_tensors.quantization import (  # noqa: E402
    QuantizationArgs,
    QuantizationScheme,
)
from compressed_tensors.quantization.quant_scheme import (  # noqa: E402
    preset_name_to_scheme,
)
from llmcompressor import oneshot  # noqa: E402
from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfm2_common import (  # noqa: E402
    CENSUS_W8A16,
    CENSUS_W8A8,
    E12_W8A16,
    E12_W8A8,
    IGNORE_SQ,
    MODELS_DIR,
    restore_expert_names,
    assert_saved_expert_names,
)


def _hard_fail_model_class():
    """Concrete class only. AutoModel* demotes config.json → vLLM rejects."""
    from transformers import Lfm2MoeForCausalLM

    return Lfm2MoeForCausalLM


def int8_channel_weights() -> QuantizationArgs:
    """INT8 per-channel symmetric weights (fresh instance per group)."""
    return QuantizationArgs(num_bits=8, type="int", symmetric=True,
                            strategy="channel")


def int8_token_dynamic_acts() -> QuantizationArgs:
    """INT8 per-token dynamic symmetric activations."""
    return QuantizationArgs(num_bits=8, type="int", symmetric=True,
                            strategy="token", dynamic=True)


def build_groups():
    group_0 = QuantizationScheme(
        targets=list(E12_W8A8),
        weights=int8_channel_weights(),
        input_activations=int8_token_dynamic_acts(),
    )
    # Prove group_0 is byte-for-byte the scheme="W8A8" preset, re-targeted.
    preset = preset_name_to_scheme("W8A8", list(E12_W8A8))
    assert group_0.weights == preset.weights, (
        f"group_0 weights differ from W8A8 preset:\n"
        f"  ours  : {group_0.weights.model_dump()}\n"
        f"  preset: {preset.weights.model_dump()}")
    assert group_0.input_activations == preset.input_activations
    assert group_0.output_activations == preset.output_activations

    group_1 = QuantizationScheme(
        targets=list(E12_W8A16),
        weights=int8_channel_weights(),
        input_activations=None,
    )
    assert group_1.weights == group_0.weights
    return {"group_0": group_0, "group_1": group_1}


def _force_moe_linearize_cpu() -> None:
    """Pin llm-compressor's MoE linearization workspace to CPU (see RTN)."""
    import llmcompressor.modeling.moe.linear_experts as _le

    _orig = _le.get_cache_init_kwargs

    def _cpu_cache(module, default=None):
        kw = dict(_orig(module, default))
        kw["onload_device"] = "cpu"
        kw["offload_device"] = "cpu"
        kw.pop("offload_dir", None)
        return kw

    _le.get_cache_init_kwargs = _cpu_cache
    print("moe linearize: converted experts pinned to CPU", flush=True)


def census(model) -> dict:
    """Count quantized Linear modules by class and by activation precision.

    Expert modules are matched on any of the three naming forms they can carry
    depending on pipeline stage: on-disk ``w1/w3/w2``, linearized
    ``gate_proj/up_proj/down_proj`` (what llm-compressor's LinearExperts2D
    produces), or a fused ``gate_up_proj`` (which would mean linearization
    never ran — reported, never silently accepted).
    """
    gate_names = (".w1", ".gate_proj")
    up_names = (".w3", ".up_proj")
    down_names = (".w2", ".down_proj")

    counts = {
        "experts_gate_a8": 0, "experts_up_a8": 0, "experts_down_a16": 0,
        "attn_a8": 0, "dense_gate_up_a8": 0, "dense_down_a16": 0,
        "conv": 0, "router": 0, "other_a8": 0, "other_a16": 0,
        "fused_experts": 0, "mismatch": [],
    }
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        scheme = getattr(mod, "quantization_scheme", None)
        if scheme is None:
            continue
        is_a8 = scheme.input_activations is not None
        short = name.replace("model.layers.", "")
        if ".experts." in name:
            if name.endswith(".gate_up_proj"):
                # Linearization never ran. Never silently accept this.
                counts["fused_experts"] += 1
            elif name.endswith(down_names):
                if is_a8:
                    counts["mismatch"].append(short)
                else:
                    counts["experts_down_a16"] += 1
            elif name.endswith(gate_names):
                if is_a8:
                    counts["experts_gate_a8"] += 1
                else:
                    counts["mismatch"].append(short)
            elif name.endswith(up_names):
                if is_a8:
                    counts["experts_up_a8"] += 1
                else:
                    counts["mismatch"].append(short)
            else:
                counts["other_a8" if is_a8 else "other_a16"] += 1
        elif ".self_attn." in name:
            if is_a8:
                counts["attn_a8"] += 1
            else:
                counts["mismatch"].append(short)
        elif ".conv." in name:
            counts["conv"] += 1
        elif name.endswith(".gate"):
            counts["router"] += 1
        elif ".feed_forward." in name:
            if name.endswith(".w2"):
                if is_a8:
                    counts["mismatch"].append(short)
                else:
                    counts["dense_down_a16"] += 1
            else:
                counts["dense_gate_up_a8" if is_a8 else "other_a16"] += 1
        else:
            counts["other_a8" if is_a8 else "other_a16"] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        MODELS_DIR, "LFM2.5-24B-A2B-smoothed"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--offload", default=None)
    ap.add_argument("--gpu-mem", type=float, default=14)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT = args.out or os.path.join(
        MODELS_DIR, "LFM2.5-24B-A2B-SQ-E12-W8A8")
    config_groups = build_groups()
    print("OK: group_0 args == W8A8 preset args", flush=True)
    for name, grp in config_groups.items():
        kind = "W8A8" if grp.input_activations is not None else "W8A16"
        print(f"{name} [{kind}] {len(grp.targets)} target(s)", flush=True)
    print(f"E12 W8A16 patterns: {E12_W8A16}", flush=True)
    print(f"ignore: {IGNORE_SQ}", flush=True)
    exp_a8 = sum(CENSUS_W8A8.values())
    exp_a16 = sum(CENSUS_W8A16.values())
    print(f"expected census: W8A8={exp_a8} W8A16={exp_a16} "
          f"total={exp_a8 + exp_a16}", flush=True)

    if args.dry_run:
        print(f"\n--dry-run: MODEL={args.model} OUT={OUT}, exiting before load",
              flush=True)
        return

    recipe = [QuantizationModifier(config_groups=config_groups,
                                   ignore=list(IGNORE_SQ))]

    ModelClass = _hard_fail_model_class()

    offload = args.offload or (OUT + ".offload")
    mm = {i: f"{args.gpu_mem:g}GiB" for i in range(torch.cuda.device_count())}
    mm["cpu"] = "40GiB"
    print(f"\nloading model (accelerate offload, max_memory={mm})...", flush=True)
    model = ModelClass.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory=mm, offload_folder=offload, offload_state_dict=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)

    _force_moe_linearize_cpu()

    oneshot(
        model=model,
        recipe=recipe,
        tokenizer=tokenizer,
        output_dir=OUT,
        trust_remote_code_model=True,
    )
    print("quantization done", flush=True)

    # ------------------------------------------------------ per-class census
    counts = census(model)
    want = {
        "experts_gate_a8": CENSUS_W8A8["experts.gate"],
        "experts_up_a8": CENSUS_W8A8["experts.up"],
        "experts_down_a16": CENSUS_W8A16["experts.down"],
        "attn_a8": CENSUS_W8A8["attn.qkvo"],
        "dense_gate_up_a8": CENSUS_W8A8["dense.gate_up"],
        "dense_down_a16": CENSUS_W8A16["dense.down"],
        "conv": 0,
        "router": 0,
    }
    print("\n=== per-class census (E12) ===", flush=True)
    bad = {}
    for k, v in want.items():
        got = counts[k]
        flag = "" if got == v else "   <-- MISMATCH"
        if got != v:
            bad[k] = (got, v)
        print(f"  {k:<20} {got:>6}  (want {v}){flag}", flush=True)
    if counts["fused_experts"]:
        bad["fused_experts"] = (counts["fused_experts"], 0)
        print(f"  FUSED EXPERTS: {counts['fused_experts']} — linearization "
              f"did not run; expert gate/up shapes are wrong for E12.",
              flush=True)
    if counts["mismatch"]:
        bad["mismatch"] = (len(counts["mismatch"]), counts["mismatch"][:5])
        print(f"  MISMODED: {counts['mismatch'][:5]}", flush=True)
    print(f"  other W8A8/W8A16: {counts['other_a8']}/{counts['other_a16']}",
          flush=True)
    if bad:
        print(f"\nWARNING: census mismatch {bad} — do NOT trust this build.",
              flush=True)
    else:
        print("\ncensus OK", flush=True)

    # NOTE: deliberately no in-process generate (see quantize_lfm2_rtn.py).
    # oneshot(output_dir=OUT) already saved; no second save_pretrained
    # (the old retry loop re-wrote ~25GB for nothing).

    n_fixed = restore_expert_names(OUT)
    print(f"expert names restored: {n_fixed} tensors -> w1/w3/w2", flush=True)
    assert_saved_expert_names(OUT)

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
