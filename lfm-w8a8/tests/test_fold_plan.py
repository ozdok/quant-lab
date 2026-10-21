"""Wiring tests for apply-smoothing-lfm2 plan_ops / apply_ops (no model load).

test_lfm2_common.py pins the fold *algebra* in isolation. These tests pin the
*wiring*: which tensors each fold touches, and that the planned ops leave the
network function unchanged. Both Path 2 failures lived in the wiring: fold A
rescaled ffn_norm but never touched the router that reads the same tensor, and
a nesting slip left fold C on expert 63 only.
"""

import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import lfm2_common as L

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts",
                      "apply-smoothing-lfm2.py")


def _load_script():
    # The script's filename has hyphens, so it cannot be imported by name.
    spec = importlib.util.spec_from_file_location("apply_smoothing_lfm2",
                                                  SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_script()


def _weight_map():
    wm = {}
    for i in L.MOE_LAYER_IDS:
        wm[f"{L.layer_prefix(i)}.feed_forward.experts.0.w1.weight"] = "shard"
    for i in L.FULL_LAYERS:
        wm[f"{L.layer_prefix(i)}.self_attn.q_proj.weight"] = "shard"
    return wm


def _scales():
    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.rand(*shape, generator=g) + 0.5

    scales = {}
    for i in L.MOE_LAYER_IDS:
        lk = L.layer_prefix(i)
        scales[f"{lk}.feed_forward.experts"] = rand(L.HIDDEN)
        scales[f"{lk}.feed_forward.experts.w2"] = rand(L.N_EXPERTS,
                                                       L.MOE_INTER)
    for i in L.FULL_LAYERS:
        lk = L.layer_prefix(i)
        scales[f"{lk}.self_attn.q_proj"] = rand(L.HIDDEN)
        scales[f"{lk}.self_attn.out_proj"] = rand(L.N_KV, L.HD)
    return scales


WM, SCALES = _weight_map(), _scales()


def _plan(folds):
    return S.plan_ops(folds, WM, SCALES)


def test_fold_a_scales_router_with_the_same_vector_as_ffn_norm():
    # The router reads ffn_norm's output exactly like the experts do, so it
    # must take the identical scale or it sees x / s_a.
    ops = _plan("A")
    for i in L.MOE_LAYER_IDS:
        lk = L.layer_prefix(i)
        gate = f"{lk}.feed_forward.gate.weight"
        assert gate in ops, f"layer {i}: router not folded"
        norm_steps = ops[f"{lk}.ffn_norm.weight"]
        assert [k for k, _ in norm_steps] == ["norm_plain"]
        assert [k for k, _ in ops[gate]] == ["colmul"], (i, ops[gate])
        assert torch.equal(norm_steps[0][1], ops[gate][0][1]), i


def test_router_logits_survive_fold_a():
    """Routing must be identical before and after the smoothing fold."""
    torch.manual_seed(0)
    i = L.MOE_LAYER_IDS[0]
    lk = L.layer_prefix(i)
    ops = _plan("A")
    x = torch.randn(32, L.HIDDEN)
    x[:, [5, 100, 1000]] *= 20.0
    norm_w = torch.rand(L.HIDDEN) + 0.5
    gate_w = torch.randn(L.N_EXPERTS, L.HIDDEN)

    def logits(norm_weight, gate_weight):
        rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        return (norm_weight * rms) @ gate_weight.T

    ref = logits(norm_w, gate_w)
    got = logits(
        S.apply_ops(norm_w, ops[f"{lk}.ffn_norm.weight"]),
        S.apply_ops(gate_w, ops.get(f"{lk}.feed_forward.gate.weight", [])))
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_fold_c_reaches_every_expert_in_every_layer():
    ops = _plan("C")
    for i in L.MOE_LAYER_IDS:
        lk = L.layer_prefix(i)
        s_c = SCALES[f"{lk}.feed_forward.experts.w2"]
        for e in range(L.N_EXPERTS):
            w3 = ops.get(f"{lk}.feed_forward.experts.{e}.w3.weight")
            w2 = ops.get(f"{lk}.feed_forward.experts.{e}.w2.weight")
            assert w3 is not None and w2 is not None, (i, e)
            assert [k for k, _ in w3] == ["rowdiv"], (i, e)
            assert torch.equal(w3[0][1], s_c[e]), (i, e)
            assert [k for k, _ in w2] == ["colmul"], (i, e)
            assert torch.equal(w2[0][1], s_c[e]), (i, e)


def test_each_fold_subset_plans_exactly_its_own_tensors():
    n_moe, n_attn, n_exp = len(L.MOE_LAYER_IDS), len(L.FULL_LAYERS), L.N_EXPERTS
    expected = {
        "A": n_moe * (2 + 2 * n_exp),      # ffn_norm + router + w1/w3 per expert
        "C": n_moe * (2 * n_exp),          # w3 rows + w2 cols per expert
        "B": n_attn * 4,                   # operator_norm + q/k/v
        "E": n_attn * 2,                   # v_proj rows + out_proj cols
        "AC": n_moe * (2 + 3 * n_exp),     # w3 is shared between A and C
        "BE": n_attn * 5,                  # v_proj is shared between B and E
        "ABCE": n_moe * (2 + 3 * n_exp) + n_attn * 5,
    }
    for folds, count in expected.items():
        assert len(_plan(folds)) == count, (folds, len(_plan(folds)), count)


def test_plan_never_touches_what_must_stay_put():
    ops = _plan("ABCE")
    banned = ("conv", "expert_bias", "embed", "layernorm")
    assert not [n for n in ops if any(b in n for b in banned)]
    # Dense layers 0-1 are plain RTN: no smoothing folds touch them.
    assert not [n for n in ops if n.startswith(("model.layers.0.",
                                                "model.layers.1."))]


def test_attention_folds_preserve_the_block_output():
    """B (q, k AND v) + E on real-shaped tensors, through a mini GQA block.

    RoPE is omitted on purpose: it rotates q/k after the folded projections
    and cannot interact with input-side or v/out folds. Also pins why the old
    'v-only' B idea is wrong: s is per-channel, so q/k need cols * s too.
    """
    torch.manual_seed(0)
    dtype = torch.float64
    i = L.FULL_LAYERS[0]
    lk = L.layer_prefix(i)
    ops = _plan("BE")
    T = 6
    x = torch.randn(T, L.HIDDEN, dtype=dtype)
    x[:, [3, 700]] *= 15.0
    qn = torch.rand(L.HD, dtype=dtype) + 0.5
    kn = torch.rand(L.HD, dtype=dtype) + 0.5
    weights = {
        "operator_norm.weight": torch.rand(L.HIDDEN, dtype=dtype) + 0.5,
        "self_attn.q_proj.weight": torch.randn(L.N_Q * L.HD, L.HIDDEN,
                                               dtype=dtype) * 0.05,
        "self_attn.k_proj.weight": torch.randn(L.N_KV * L.HD, L.HIDDEN,
                                               dtype=dtype) * 0.05,
        "self_attn.v_proj.weight": torch.randn(L.N_KV * L.HD, L.HIDDEN,
                                               dtype=dtype) * 0.05,
        "self_attn.out_proj.weight": torch.randn(L.HIDDEN, L.N_Q * L.HD,
                                                 dtype=dtype) * 0.05,
    }

    def rms(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5)

    def block(w):
        y = w["operator_norm.weight"] * rms(x)
        q = rms((y @ w["self_attn.q_proj.weight"].T)
                .view(T, L.N_Q, L.HD)) * qn
        k = rms((y @ w["self_attn.k_proj.weight"].T)
                .view(T, L.N_KV, L.HD)) * kn
        v = (y @ w["self_attn.v_proj.weight"].T).view(T, L.N_KV, L.HD)
        k = k.repeat_interleave(L.G, dim=1)
        v = v.repeat_interleave(L.G, dim=1)
        att = torch.einsum("qhd,khd->hqk", q, k) / L.HD ** 0.5
        causal = torch.ones(T, T).triu(1).bool()
        att = att.masked_fill(causal, float("-inf")).softmax(-1)
        out = torch.einsum("hqk,khd->qhd", att, v).reshape(T, L.N_Q * L.HD)
        return out @ w["self_attn.out_proj.weight"].T

    def folded(plan):
        return {n: S.apply_ops(w, plan.get(f"{lk}.{n}", []))
                for n, w in weights.items()}

    ref = block(weights)
    torch.testing.assert_close(block(folded(ops)), ref, rtol=1e-7, atol=1e-9)

    v_only = {n: s for n, s in ops.items()
              if not n.endswith(("q_proj.weight", "k_proj.weight"))}
    rel = ((block(folded(v_only)) - ref).norm() / ref.norm()).item()
    assert rel > 1e-3, f"v-only B should be inexact (rel {rel:.2e})"
