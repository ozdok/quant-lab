"""Tests for lfm2_common geometry + recipe tables (no heavy imports)."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import lfm2_common as L


def test_layer_split():
    assert L.N_LAYERS == 40
    assert L.N_ATTN == 10
    assert L.N_CONV == 30
    # LFM2 puts full attention at 2,6,...,38 — NOT at 3,7,...,39.
    assert L.FULL_LAYERS == list(range(2, 40, 4))
    assert all(L.is_full_attn_layer(i) for i in (2, 6, 38))
    assert not L.is_full_attn_layer(0)
    assert not L.is_full_attn_layer(3)


def test_moe_counts():
    assert L.N_EXPERTS == 64
    assert L.TOPK == 4
    assert L.MOE_INTER == 1536
    assert L.G == 4  # 32Q / 8KV
    assert L.N_KV == 8 and L.HD == 64
    # No shared expert in this model.
    assert L.N_SHARED == 0
    assert L.DENSE_LAYERS == 2 and L.DENSE_INTER == 11776
    assert L.N_MOE_LAYERS == 38
    assert L.is_moe_layer(2) and not L.is_moe_layer(1)


def test_layer_prefix_has_no_language_model_wrapper():
    # The single most common path mistake: other families use
    # `model.language_model.layers.N`; LFM2 uses `model.layers.N`.
    assert L.layer_prefix(0) == "model.layers.0"
    assert L.layer_prefix(39) == "model.layers.39"
    assert "language_model" not in L.layer_prefix(0)


def test_router_pattern_matches_lfm2_not_the_mlp_name():
    pats = L.IGNORE_RTN
    gate_pat = next(p for p in pats if p.endswith(r"\.gate$"))

    # The real LFM2 router must be ignored.
    assert L.matches("model.layers.2.feed_forward.gate", gate_pat)

    # Regression guard: the `mlp.gate` pattern (`re:.*mlp[.]gate$`) does NOT
    # match LFM2's router. If someone reverts to it, the router gets
    # quantized.
    mlp_pat = r"re:.*mlp[.]gate$"
    assert not L.matches("model.layers.2.feed_forward.gate", mlp_pat)

    # ...and the LFM2 pattern must not catch expert weights.
    assert not L.matches("model.layers.2.feed_forward.experts.0.w1", gate_pat)
    assert not L.matches("model.layers.2.feed_forward.experts.5.w3", gate_pat)


def test_conv_block_ignored():
    conv_pat = next(p for p in L.IGNORE_RTN if "conv" in p)
    for name in ("model.layers.3.conv.in_proj",
                 "model.layers.3.conv.out_proj",
                 "model.layers.3.conv.conv",
                 "model.layers.39.conv.in_proj"):
        assert L.matches(name, conv_pat), name
    # The `linear_attn` blanket pattern other families use matches nothing
    # here — another silent wiring bug.
    linear_attn_pat = r"re:.*linear_attn.*"
    assert not L.matches("model.layers.3.conv.in_proj", linear_attn_pat)


def test_rtn_keeps_experts_quantizable():
    """No IGNORE_RTN pattern may catch an expert, attn, or dense Linear."""
    quantizable = [
        "model.layers.2.feed_forward.experts.0.w1",
        "model.layers.2.feed_forward.experts.63.w3",
        "model.layers.2.feed_forward.experts.17.w2",
        "model.layers.2.self_attn.q_proj",
        "model.layers.38.self_attn.out_proj",
        "model.layers.0.feed_forward.w1",
        "model.layers.1.feed_forward.w2",
    ]
    for name in quantizable:
        for pat in L.IGNORE_RTN:
            assert not L.matches(name, pat), f"{name} wrongly ignored by {pat}"


def test_e12_patterns_split_gate_up_from_down():
    gate = r"re:.*[.]experts[.].*w1$"
    up = r"re:.*[.]experts[.].*w3$"
    down = r"re:.*[.]experts[.].*w2$"

    assert any(p == gate for p in L.E12_W8A8)
    assert any(p == up for p in L.E12_W8A8)
    assert any(p == down for p in L.E12_W8A16)

    for proj, suffix, a8 in ((gate, "w1", True),
                             (up, "w3", True),
                             (down, "w2", False)):
        name = f"model.layers.2.feed_forward.experts.7.{suffix}"
        assert L.matches(name, proj), name
        assert proj in (L.E12_W8A8 if a8 else L.E12_W8A16)

    # No cross-talk between the three expert projections.
    assert not L.matches("model.layers.2.feed_forward.experts.7.w2", gate)
    assert not L.matches("model.layers.2.feed_forward.experts.7.w2", up)
    assert not L.matches("model.layers.2.feed_forward.experts.7.w1", down)
    assert not L.matches("model.layers.2.feed_forward.experts.7.w3", down)


def test_e12_dense_patterns_do_not_touch_experts():
    dense_gate = r"re:.*[.]feed_forward[.]w1$"
    dense_down = r"re:.*[.]feed_forward[.]w2$"

    assert L.matches("model.layers.0.feed_forward.w1", dense_gate)
    assert L.matches("model.layers.1.feed_forward.w2", dense_down)

    # Must not reach into the MoE expert tensors of MoE layers.
    assert not L.matches("model.layers.2.feed_forward.experts.0.w1", dense_gate)
    assert not L.matches("model.layers.2.feed_forward.experts.0.w2", dense_down)
    # Nor the router.
    assert not L.matches("model.layers.2.feed_forward.gate", dense_gate)


def test_e12_handles_fused_expansion_names():
    """If llm-compressor fuses during oneshot, the fused names still split."""
    fused_gu = "model.layers.2.feed_forward.experts.gate_up_proj"
    fused_d = "model.layers.2.feed_forward.experts.down_proj"
    assert any(L.matches(fused_gu, p) for p in L.E12_W8A8)
    assert any(L.matches(fused_d, p) for p in L.E12_W8A16)
    assert not any(L.matches(fused_d, p) for p in L.E12_W8A8)


def test_e12_covers_linearized_expert_names():
    """llm-compressor's LinearExperts2D renames experts gate/up/down_proj.

    Verified against the installed
    llmcompressor/modeling/moe/linear_experts.py. These names must be covered
    or the expert FFNs — 96% of the model — silently escape E12 targeting and
    stay BF16.
    """
    for e in (0, 7, 63):
        gate = f"model.layers.2.feed_forward.experts.{e}.gate_proj"
        up = f"model.layers.2.feed_forward.experts.{e}.up_proj"
        down = f"model.layers.2.feed_forward.experts.{e}.down_proj"
        assert any(L.matches(gate, p) for p in L.E12_W8A8), gate
        assert any(L.matches(up, p) for p in L.E12_W8A8), up
        assert any(L.matches(down, p) for p in L.E12_W8A16), down
        # and never the wrong group
        assert not any(L.matches(down, p) for p in L.E12_W8A8), down
        assert not any(L.matches(gate, p) for p in L.E12_W8A16), gate
        assert not any(L.matches(up, p) for p in L.E12_W8A16), up


def test_router_ignore_does_not_catch_linearized_gate_proj():
    """The router ignore is `re:.*\\.gate$`.

    After linearization expert gates are named `...experts.{e}.gate_proj`,
    which must NOT end in `.gate`. If it did, the router pattern would eat
    2432 expert gate tensors in Path 1 and silently leave them BF16.
    """
    router_pat = next(p for p in L.IGNORE_RTN if p.endswith(r"\.gate$"))
    assert L.matches("model.layers.2.feed_forward.gate", router_pat)
    for e in (0, 7, 63):
        assert not L.matches(
            f"model.layers.2.feed_forward.experts.{e}.gate_proj", router_pat)
        assert not L.matches(
            f"model.layers.2.feed_forward.experts.{e}.up_proj", router_pat)
        assert not L.matches(
            f"model.layers.2.feed_forward.experts.{e}.down_proj", router_pat)


def test_rtn_ignore_safe_on_linearized_names():
    """Every post-linearization expert/attn/dense name must stay quantizable."""
    names = []
    for e in (0, 7, 63):
        for p in ("gate_proj", "up_proj", "down_proj"):
            names.append(
                f"model.layers.2.feed_forward.experts.{e}.{p}")
    for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
        names.append(f"model.layers.2.self_attn.{p}")
    for p in ("w1", "w2", "w3"):
        names.append(f"model.layers.1.feed_forward.{p}")
    for name in names:
        for pat in L.IGNORE_RTN:
            assert not L.matches(name, pat), f"{name} wrongly ignored by {pat}"


def test_census_sums():
    assert L.CENSUS_W8A8["experts.gate"] == 64 * 38
    assert L.CENSUS_W8A8["experts.up"] == 64 * 38
    assert L.CENSUS_W8A16["experts.down"] == 64 * 38
    assert L.CENSUS_W8A8["attn.qkvo"] == 10 * 4
    assert L.CENSUS_W8A8["dense.gate_up"] == 2 * 2
    assert L.CENSUS_W8A16["dense.down"] == 2
    # Path 1 quantizes every expert/attn/dense Linear, nothing else.
    assert L.CENSUS_RTN_TOTAL == 64 * 38 * 3 + 10 * 4 + 2 * 3 == 7342
    assert sum(L.CENSUS_W8A8.values()) + sum(L.CENSUS_W8A16.values()) == 7342


def test_path4_patterns_cover_all_expert_forms_and_keep_attn_dense_w8a16():
    """Path 4 (W8A8-EXPERTS): all 3 expert projections (on-disk, linearized,
    and fused) go in PATH4_W8A8; attention q/k/v/out and dense layers 0-1 w1/w2/w3
    go in PATH4_W8A16; no cross-talk between the two groups."""
    for e in (0, 7, 63):
        for suffix in ("w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.2.feed_forward.experts.{e}.{suffix}"
            assert any(L.matches(name, p) for p in L.PATH4_W8A8), name
            assert not any(L.matches(name, p) for p in L.PATH4_W8A16), name
    for fused in ("gate_up_proj", "down_proj"):
        name = f"model.layers.2.feed_forward.experts.{fused}"
        assert any(L.matches(name, p) for p in L.PATH4_W8A8), name
        assert not any(L.matches(name, p) for p in L.PATH4_W8A16), name

    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        name = f"model.layers.2.self_attn.{proj}"
        assert any(L.matches(name, p) for p in L.PATH4_W8A16), name
        assert not any(L.matches(name, p) for p in L.PATH4_W8A8), name

    for layer in (0, 1):
        for proj in ("w1", "w2", "w3"):
            name = f"model.layers.{layer}.feed_forward.{proj}"
            assert any(L.matches(name, p) for p in L.PATH4_W8A16), name
            assert not any(L.matches(name, p) for p in L.PATH4_W8A8), name

    # Router and conv must not match either target list (and are in IGNORE_RTN).
    for kept in ("model.layers.2.feed_forward.gate",
                 "model.layers.3.conv.in_proj",
                 "model.layers.3.conv.out_proj"):
        assert not any(L.matches(kept, p) for p in L.PATH4_W8A8), kept
        assert not any(L.matches(kept, p) for p in L.PATH4_W8A16), kept


def test_path4_census_sums():
    assert L.CENSUS_PATH4_W8A8 == {"experts": 64 * 38 * 3}
    assert L.CENSUS_PATH4_W8A16 == {"attn": 10 * 4, "dense": 2 * 3}
    assert (sum(L.CENSUS_PATH4_W8A8.values())
            + sum(L.CENSUS_PATH4_W8A16.values())) == L.CENSUS_RTN_TOTAL == 7342


def test_up_rowdiv_only_is_exact_for_swiglu():
    """LFM2's experts are separate w1/w3/w2 tensors, so the moe-down fold
    scales w3 (up) rows only — which is exact for silu(gate) * up.

    A fused-halves C-fold divides BOTH fused halves (gate and up), which
    is NOT exact because the gate side passes through SiLU first. This
    test pins the correct behaviour and would catch a regression to the
    fused-halves approach.
    """
    torch.manual_seed(0)
    I, H, T = 8, 16, 5
    x = torch.randn(T, H)
    w1 = torch.randn(I, H)   # gate
    w3 = torch.randn(I, H)   # up
    w2 = torch.randn(H, I)   # down
    s = torch.rand(I) + 0.5

    gate = torch.nn.functional.silu(x @ w1.T)
    up = x @ w3.T

    # Correct (LFM2): divide up rows by s, then multiply down cols by s.
    ref = (gate * up) @ w2.T
    got = (gate * (up / s)) @ (w2 * s.unsqueeze(0)).T
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()

    # Wrong (fused-halves bug): also scale the gate side.
    wrong = (torch.nn.functional.silu(x @ w1.T / s) * (up / s)) @ \
            (w2 * s.unsqueeze(0)).T
    assert not torch.allclose(wrong, ref, atol=1e-3), \
        "expected the fused-halves fold to be inexact — the test is not " \
        "actually exercising the bug"


def test_colmul_broadcasts_over_input_dim():
    # s_a [H] must scale the INPUT (last) dim of a [out, H] weight.
    w = torch.ones(4, 6)
    s = torch.arange(6).float() + 1
    got = w * s.unsqueeze(0)
    assert got.shape == (4, 6)
    assert torch.equal(got[0], s)


def test_regex_for():
    assert L.regex_for("feed_forward.experts.down_proj") == \
        r"re:.*\.feed_forward\.experts\.down_proj$"


# ------------------------------------------------------------------- smoke
def _smoke_dtypes(*, experts_quantized=True, router_quantized=False,
                  foreign_family=False):
    """Build a {name: dtype} map shaped like the real RTN output."""
    d = {
        "model.embed_tokens.weight": "BF16",
        "model.embedding_norm.weight": "BF16",
        "lm_head.weight": "BF16",
    }
    for i in range(L.N_LAYERS):
        d[f"model.layers.{i}.operator_norm.weight"] = "BF16"
        d[f"model.layers.{i}.ffn_norm.weight"] = "BF16"
        if i in L.FULL_LAYERS:
            for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
                d[f"model.layers.{i}.self_attn.{p}.weight"] = "I8"
                d[f"model.layers.{i}.self_attn.{p}.weight_scale"] = "F32"
            d[f"model.layers.{i}.self_attn.q_layernorm.weight"] = "BF16"
            d[f"model.layers.{i}.self_attn.k_layernorm.weight"] = "BF16"
        else:
            for p in ("conv", "in_proj", "out_proj"):
                d[f"model.layers.{i}.conv.{p}.weight"] = "BF16"
        if L.is_moe_layer(i):
            d[f"model.layers.{i}.feed_forward.gate.weight"] = (
                "I8" if router_quantized else "BF16")
            d[f"model.layers.{i}.feed_forward.expert_bias"] = "F32"
            for e in range(L.N_EXPERTS):
                for p in ("w1", "w2", "w3"):
                    n = f"model.layers.{i}.feed_forward.experts.{e}.{p}"
                    d[f"{n}.weight"] = "I8" if experts_quantized else "BF16"
                    d[f"{n}.weight_scale"] = "F32"
        else:
            for p in ("w1", "w2", "w3"):
                n = f"model.layers.{i}.feed_forward.{p}"
                d[f"{n}.weight"] = "I8"
                d[f"{n}.weight_scale"] = "F32"
    if foreign_family:
        d["model.layers.2.mlp.shared_expert.gate_proj.weight"] = "I8"
    return d


def test_smoke_analyze_does_not_count_scales_as_unquantized_experts():
    """Regression: expert `weight_scale` siblings must not be counted as
    unquantized expert weights.

    Counting them made a correct 7296-expert build report 7296 "left
    unquantized" and fail its own smoke check.
    """
    import smoke

    rep = smoke.analyze(_smoke_dtypes())
    assert len(rep["expert_int8"]) == L.CENSUS_RTN_TOTAL - L.N_ATTN * 4 - 6
    assert len(rep["expert_int8"]) == 7296
    assert len(rep["expert_all"]) == 7296 * 2, "weights + scales"
    assert len(rep["expert_weights"]) == 7296
    smoke.check(rep, expect_experts=7296)  # must not raise


def test_smoke_analyze_accepts_pack_quantized_w8a16_layout():
    """Path 3 (W8A16-CHANNEL) stores weights as .weight_packed (I32) +
    .weight_scale + .weight_shape (I64); smoke.analyze/check must accept both
    int-quantized (.weight I8) and pack-quantized (.weight_packed I32)."""
    import smoke

    base = _smoke_dtypes()
    packed = {}
    for k, v in base.items():
        if k.endswith(".weight") and v == "I8":
            stem = k[:-len(".weight")]
            packed[f"{stem}.weight_packed"] = "I32"
            packed[f"{stem}.weight_shape"] = "I64"
        else:
            packed[k] = v
    rep = smoke.analyze(packed)
    assert len(rep["expert_weights"]) == 7296
    assert len(rep["expert_int8"]) == 7296
    assert len(rep["int8"]) == 7342
    smoke.check(rep, expect_experts=7296)


def test_smoke_check_rejects_unquantized_experts():
    import smoke

    rep = smoke.analyze(_smoke_dtypes(experts_quantized=False))
    try:
        smoke.check(rep)
    except AssertionError as exc:
        assert "NOT quantized" in str(exc) or "unquantized" in str(exc)
    else:
        raise AssertionError("smoke accepted a build with BF16 experts")


def test_smoke_check_rejects_quantized_router():
    import smoke

    rep = smoke.analyze(_smoke_dtypes(router_quantized=True))
    try:
        smoke.check(rep)
    except AssertionError as exc:
        assert "router" in str(exc)
    else:
        raise AssertionError("smoke accepted a quantized router")


def test_smoke_check_rejects_foreign_family_tensors():
    import smoke

    rep = smoke.analyze(_smoke_dtypes(foreign_family=True))
    try:
        smoke.check(rep)
    except AssertionError as exc:
        assert "foreign-family" in str(exc)
    else:
        raise AssertionError("smoke accepted foreign-family tensors")


# ---------------------------------------------------------------- preflight
def _write_index(tmp_path, *, n_experts=64, attn_at=range(2, 40, 4), extra=None):
    """Synthesise a safetensors index for the LFM2 geometry."""
    wm = {"model.embed_tokens.weight": "model-00001-of-00001.safetensors",
          "model.embedding_norm.weight": "model-00001-of-00001.safetensors"}
    attn_set = set(attn_at)
    for i in range(L.N_LAYERS):
        shard = "model-00001-of-00001.safetensors"
        wm[f"{L.layer_prefix(i)}.operator_norm.weight"] = shard
        wm[f"{L.layer_prefix(i)}.ffn_norm.weight"] = shard
        if i in attn_set:
            for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
                wm[f"{L.layer_prefix(i)}.self_attn.{p}.weight"] = shard
        else:
            wm[f"{L.layer_prefix(i)}.conv.conv.weight"] = shard
            wm[f"{L.layer_prefix(i)}.conv.in_proj.weight"] = shard
            wm[f"{L.layer_prefix(i)}.conv.out_proj.weight"] = shard
        if L.is_moe_layer(i):
            wm[f"{L.layer_prefix(i)}.feed_forward.gate.weight"] = shard
            wm[f"{L.layer_prefix(i)}.feed_forward.expert_bias"] = shard
            for e in range(n_experts):
                for p in ("w1", "w2", "w3"):
                    wm[f"{L.layer_prefix(i)}.feed_forward.experts.{e}.{p}.weight"] = shard
        else:
            for p in ("w1", "w2", "w3"):
                wm[f"{L.layer_prefix(i)}.feed_forward.{p}.weight"] = shard
    for name in (extra or []):
        wm[name] = "model-00001-of-00001.safetensors"
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wm}))
    return tmp_path


def test_preflight_accepts_real_geometry(tmp_path):
    import quantize_lfm2_rtn as Q

    src = _write_index(tmp_path / "ok")
    info = Q.preflight_geometry(str(src))
    assert info["layers"] == 40
    assert info["attn_idx"] == [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]
    assert len(info["conv_idx"]) == 30
    assert info["moe_layers"] == 38
    assert info["dense_layers"] == 2
    assert info["experts"] == 64


def test_preflight_rejects_wrong_attention_parity(tmp_path):
    """Attention on 3,7,...,39 must not pass."""
    import quantize_lfm2_rtn as Q

    src = _write_index(tmp_path / "other_parity", attn_at=range(3, 40, 4))
    try:
        Q.preflight_geometry(str(src))
    except SystemExit as exc:
        msg = str(exc)
        assert "attention layer indices" in msg and "short-conv layer indices" in msg
    else:
        raise AssertionError("preflight accepted wrong attention parity")


def test_preflight_rejects_wrong_expert_count(tmp_path):
    import quantize_lfm2_rtn as Q

    src = _write_index(tmp_path / "wrong_experts", n_experts=256)
    try:
        Q.preflight_geometry(str(src))
    except SystemExit as exc:
        assert "experts 256 != 64" in str(exc)
    else:
        raise AssertionError("preflight accepted 256 experts")


def test_preflight_rejects_foreign_family_tensors(tmp_path):
    import quantize_lfm2_rtn as Q

    src = _write_index(tmp_path / "foreign_tensors",
                       extra=["model.layers.2.mlp.shared_expert.gate_proj.weight"])
    try:
        Q.preflight_geometry(str(src))
    except SystemExit as exc:
        assert "shared_expert" in str(exc)
    else:
        raise AssertionError("preflight accepted foreign-family tensors")

def test_linearized_names_restore_to_w123():
    assert L.fixed_expert_name(
        "model.layers.2.feed_forward.experts.0.gate_proj.weight") ==         "model.layers.2.feed_forward.experts.0.w1.weight"
    assert L.fixed_expert_name(
        "model.layers.2.feed_forward.experts.7.up_proj.weight_scale") ==         "model.layers.2.feed_forward.experts.7.w3.weight_scale"
    assert L.fixed_expert_name(
        "model.layers.2.feed_forward.experts.63.down_proj.weight") ==         "model.layers.2.feed_forward.experts.63.w2.weight"
    assert L.fixed_expert_name(
        "model.layers.0.feed_forward.w1.weight") ==         "model.layers.0.feed_forward.w1.weight"
    assert L.fixed_expert_name(
        "model.layers.2.feed_forward.gate.weight") ==         "model.layers.2.feed_forward.gate.weight"


def test_e12_linearized_forms_still_split():
    assert L.matches("model.layers.2.feed_forward.experts.7.gate_proj",
                     "re:.*[.]experts[.].*gate_proj$")
    assert L.matches("model.layers.2.feed_forward.experts.7.up_proj",
                     "re:.*[.]experts[.].*up_proj$")
