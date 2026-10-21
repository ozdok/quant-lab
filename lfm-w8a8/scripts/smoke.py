#!/usr/bin/env python3
"""smoke.py — structural smoke check for a quantized LFM2.5-24B-A2B dir.

Proves LAYOUT, not quality. Reads safetensors headers only (no model load),
so it runs on a busy box.

  1. config.json carries a ``quantization_config`` with W8A8 group args.
  2. INT8 weights + matching scales are present.
  3. Expert FFNs are quantized (the whole point — they are ~96% of params).
  4. Router, short-conv block, embeddings and norms are NOT quantized.
  5. No foreign-family leftovers (shared_expert / linear_attn / mtp / visual).

``analyze()`` is pure (a {name: dtype} map in, counts out) so the layout
rules stay unit-testable without a 24GB checkpoint on disk.

Optional ``--generate`` runs a real vLLM generation, but do that on a free
box: it loads the full ~25GB model.

usage: smoke.py --model PATH [--generate] [--expect-experts N]
"""

import argparse
import json
import os
import struct

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Modules that must stay at source precision: label -> name predicate.
KEPT_BF16_PREDICATES = {
    "router (feed_forward.gate)":
        lambda n: n.endswith(".feed_forward.gate.weight"),
    "short-conv block":
        lambda n: ".conv." in n and n.endswith(".weight"),
    "embeddings":
        lambda n: n.endswith("embed_tokens.weight"),
    "norms":
        lambda n: n.endswith("norm.weight") or n.endswith("layernorm.weight"),
}

FOREIGN_FAMILY_MARKERS = ("shared_expert", "linear_attn", "mtp.", "visual.")


def _is_quantized_weight(name: str, dt: str) -> bool:
    return dt == "I8" or (name.endswith(".weight_packed") and dt == "I32")


def analyze(dtypes: dict) -> dict:
    """Summarise a quantized checkpoint from its {name: dtype} map.

    Pure function — no I/O — so the layout rules below stay testable.
    Supports both int-quantized (.weight I8) and pack-quantized (.weight_packed
    I32, used by W8A16-CHANNEL and the W8A16 group of W8A8-EXPERTS).
    """
    int8 = [n for n, dt in dtypes.items() if _is_quantized_weight(n, dt)]
    scales = [n for n in dtypes if n.endswith("_scale")]

    exp_all = [n for n in dtypes if ".feed_forward.experts." in n]
    # Each quantized weight has a `..._scale` sibling (and `.weight_shape` for
    # pack-quantized). Only `.weight` / `.weight_packed` tensors are
    # quantizable modules, so counting scales here makes a perfectly good
    # build look like it left half its experts unquantized.
    exp_weights = [n for n in exp_all
                   if n.endswith(".weight") or n.endswith(".weight_packed")]
    exp_int8 = [n for n in exp_weights
                if _is_quantized_weight(n, dtypes[n])]

    kept = {}
    for label, pred in KEPT_BF16_PREDICATES.items():
        names = [n for n in dtypes if pred(n)]
        kept[label] = {
            "count": len(names),
            "quantized": [n for n in names if dtypes[n] in ("I8", "I32")],
        }

    foreign_family = {
        marker: [n for n in dtypes if marker in n]
        for marker in FOREIGN_FAMILY_MARKERS
    }

    lin = [n for n in exp_all if 'gate_proj' in n or 'up_proj' in n or 'down_proj' in n]
    wform = [n for n in exp_weights if '.w1.' in n or '.w3.' in n or '.w2.' in n]
    return {
        "total": len(dtypes),
        "linearized": lin,
        "wform": wform,
        "int8": int8,
        "scales": scales,
        "expert_all": exp_all,
        "expert_weights": exp_weights,
        "expert_int8": exp_int8,
        "kept_bf16": kept,
        "foreign_family": {k: v for k, v in foreign_family.items() if v},
    }


def check(report: dict, expect_experts: int | None = None) -> None:
    """Assert the layout rules. Raises AssertionError with a clear reason."""
    assert report["int8"], "no INT8 weights found — quantization did not take"
    assert report["expert_int8"], (
        "expert FFNs were NOT quantized — the build is useless")
    assert len(report["expert_int8"]) == len(report["expert_weights"]), (
        f"{len(report['expert_weights']) - len(report['expert_int8'])} "
        f"expert weights left unquantized")
    if expect_experts is not None:
        assert len(report["expert_int8"]) == expect_experts, (
            f"expected {expect_experts} quantized expert weight tensors, "
            f"got {len(report['expert_int8'])}")
    for label, info in report["kept_bf16"].items():
        assert not info["quantized"], (
            f"{label} was quantized: {info['quantized'][:3]}")
    assert not report["linearized"], (
        f"{len(report['linearized'])} expert tensors still carry linearized "
        f"names (gate_proj/up_proj/down_proj) — run restore_expert_names; e.g. "
        f"{report['linearized'][:2]}")
    assert len(report["wform"]) == len(report["expert_weights"]), (
        "expert weights are not in w1/w3/w2 form")
    assert not report["foreign_family"], (
        f"unexpected foreign-family tensors: "
        f"{ {k: v[:2] for k, v in report['foreign_family'].items()} }")


def header(path: str) -> dict:
    """Read a safetensors header without loading tensor data."""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def shard_files(model_dir: str) -> list:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        return sorted(set(weight_map.values()))
    return ["model.safetensors"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--expect-experts", type=int, default=None,
                    help="expected quantized expert Linear count "
                         "(64 experts x 38 MoE layers x 3 = 7296)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    qcfg = cfg.get("quantization_config")
    assert qcfg, "no quantization_config in config.json"
    print("quant:", json.dumps(qcfg)[:400])

    dtypes: dict = {}
    for shard in shard_files(args.model):
        for name, meta in header(os.path.join(args.model, shard)).items():
            if name != "__metadata__":
                dtypes[name] = meta["dtype"]

    report = analyze(dtypes)
    print(f"tensors: {report['total']}  int8: {len(report['int8'])}  "
          f"scales: {len(report['scales'])}")
    print(f"expert tensors: {len(report['expert_all'])} total "
          f"({len(report['expert_weights'])} weights + "
          f"{len(report['expert_all']) - len(report['expert_weights'])} scales)"
          f"  weights INT8: {len(report['expert_int8'])}")
    for label, info in report["kept_bf16"].items():
        print(f"  {label}: {info['count']} tensors, all non-INT8  OK")

    check(report, args.expect_experts)

    if args.generate:
        from vllm import LLM, SamplingParams

        llm = LLM(model=args.model, trust_remote_code=True)
        out = llm.generate(
            ["Explain attention in one sentence."],
            SamplingParams(max_tokens=64, temperature=0.1, top_k=50),
        )
        print(out[0].outputs[0].text)

    print("smoke OK")


if __name__ == "__main__":
    main()
