"""Tests for RTN scheme selection: shipped W8A8 vs the W8A16-CHANNEL probe.

The probe splits Path 1's KLD gap into weight rounding versus activation
quantization. That only works if it keeps Path 1's weight quantization
exactly: compressed-tensors 0.18's stock "W8A16" preset is group-128, finer
than the per-channel weights Path 1 ships, and would flatter the weight
error. These tests pin that, and the guard that keeps a probe build from
ever overwriting the shipped W8A8 directory (``--dst`` defaults to it).
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import lfm2_common as L

PROBE = "W8A16-CHANNEL"
PATH4 = "W8A8-EXPERTS"


def _write_config(model_dir, weights_overrides=None, with_inputs=True):
    weights = {"num_bits": 8, "type": "int", "symmetric": True,
               "strategy": "channel", "group_size": None, "dynamic": False}
    weights.update(weights_overrides or {})
    inputs = ({"num_bits": 8, "type": "int", "symmetric": True,
               "strategy": "token", "dynamic": True} if with_inputs else None)
    config = {"quantization_config": {"config_groups": {"group_0": {
        "targets": ["Linear"], "weights": weights,
        "input_activations": inputs}}}}
    (model_dir / "config.json").write_text(json.dumps(config))
    return str(model_dir)


def _write_path4_config(model_dir, g0_weights_overrides=None,
                        g1_weights_overrides=None,
                        g0_inputs=True, g1_inputs=False,
                        g0_targets=None, g1_targets=None):
    base_w = {"num_bits": 8, "type": "int", "symmetric": True,
              "strategy": "channel", "group_size": None, "dynamic": False}
    w0 = dict(base_w, **(g0_weights_overrides or {}))
    w1 = dict(base_w, **(g1_weights_overrides or {}))
    act = {"num_bits": 8, "type": "int", "symmetric": True,
           "strategy": "token", "dynamic": True}
    config = {"quantization_config": {"config_groups": {
        "group_0": {
            "targets": list(L.PATH4_W8A8 if g0_targets is None else g0_targets),
            "weights": w0,
            "input_activations": act if g0_inputs else None,
        },
        "group_1": {
            "targets": list(L.PATH4_W8A16 if g1_targets is None else g1_targets),
            "weights": w1,
            "input_activations": act if g1_inputs else None,
        },
    }}}
    (model_dir / "config.json").write_text(json.dumps(config))
    return str(model_dir)


# ---- default destination + overwrite guard ------------------------------

def test_w8a8_default_dst_is_still_the_shipped_directory():
    assert L.default_dst_for("W8A8") == L.DEFAULT_DST


def test_probe_default_dst_is_a_different_directory():
    dst = L.default_dst_for(PROBE)
    assert dst != L.DEFAULT_DST
    assert os.path.basename(dst) == f"{L.DEFAULT_MODEL_NAME}-{PROBE}"


def test_path4_default_dst_is_its_own_directory():
    dst = L.default_dst_for(PATH4)
    assert dst != L.DEFAULT_DST
    assert dst != L.default_dst_for(PROBE)
    assert os.path.basename(dst) == f"{L.DEFAULT_MODEL_NAME}-{PATH4}"
    with pytest.raises(SystemExit):
        L.assert_not_shipped_dst(PATH4, L.DEFAULT_DST)


def test_unknown_scheme_is_rejected_everywhere():
    with pytest.raises(ValueError):
        L.default_dst_for("W4A16")
    with pytest.raises(ValueError):
        L.assert_not_shipped_dst("W4A16", "/tmp/x")
    with pytest.raises(ValueError):
        L.assert_saved_scheme("/tmp/does-not-matter", "W4A16")


@pytest.mark.parametrize("spelling", [
    lambda p: p,
    lambda p: p + os.sep,
    lambda p: os.path.join(p, "..", os.path.basename(p)),
])
def test_probe_cannot_target_the_shipped_directory(spelling):
    with pytest.raises(SystemExit):
        L.assert_not_shipped_dst(PROBE, spelling(L.DEFAULT_DST))


def test_w8a8_may_still_rebuild_the_shipped_directory():
    # Path 1 rebuilds in place (the 2026-10-15 rename fix did exactly this).
    L.assert_not_shipped_dst("W8A8", L.DEFAULT_DST)


def test_probe_may_target_any_other_directory(tmp_path):
    L.assert_not_shipped_dst(PROBE, str(tmp_path / "probe"))


# ---- saved-checkpoint scheme verification -------------------------------

def test_saved_scheme_accepts_matching_configs(tmp_path):
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    L.assert_saved_scheme(_write_config(shipped, with_inputs=True), "W8A8")
    probe = tmp_path / "probe"
    probe.mkdir()
    L.assert_saved_scheme(_write_config(probe, with_inputs=False), PROBE)
    path4 = tmp_path / "path4"
    path4.mkdir()
    L.assert_saved_scheme(_write_path4_config(path4), PATH4)
    # ModelCompressor writes group_0 = attn/dense (layers.0 comes first) and
    # group_1 = experts (layers.2 comes later); both group orderings must pass.
    path4_swapped = tmp_path / "path4_swapped"
    path4_swapped.mkdir()
    L.assert_saved_scheme(
        _write_path4_config(
            path4_swapped,
            g0_inputs=False, g1_inputs=True,
            g0_targets=L.PATH4_W8A16, g1_targets=L.PATH4_W8A8,
        ),
        PATH4,
    )


def test_saved_scheme_validates_path4_two_group_split(tmp_path):
    # Path 4 must have group_0 = W8A8 on experts and group_1 = W8A16 on attn+dense.
    flipped = tmp_path / "flipped"
    flipped.mkdir()
    _write_path4_config(flipped, g0_inputs=False, g1_inputs=True)
    with pytest.raises(AssertionError):
        L.assert_saved_scheme(str(flipped), PATH4)

    all_a8 = tmp_path / "all_a8"
    all_a8.mkdir()
    _write_path4_config(all_a8, g0_inputs=True, g1_inputs=True)
    with pytest.raises(AssertionError):
        L.assert_saved_scheme(str(all_a8), PATH4)

    bad_g1_weights = tmp_path / "bad_g1_weights"
    bad_g1_weights.mkdir()
    _write_path4_config(bad_g1_weights,
                        g1_weights_overrides={"strategy": "group", "group_size": 128})
    with pytest.raises(AssertionError):
        L.assert_saved_scheme(str(bad_g1_weights), PATH4)

    single_group = tmp_path / "single"
    single_group.mkdir()
    _write_config(single_group, with_inputs=True)
    with pytest.raises(AssertionError):
        L.assert_saved_scheme(str(single_group), PATH4)


def test_saved_scheme_rejects_an_activation_mismatch(tmp_path):
    # A probe that silently kept the activation quantizer measures nothing;
    # a shipped build that lost it would serve as W8A16.
    with_inputs = tmp_path / "with_inputs"
    with_inputs.mkdir()
    _write_config(with_inputs, with_inputs=True)
    with pytest.raises(AssertionError):
        L.assert_saved_scheme(str(with_inputs), PROBE)
    without_inputs = tmp_path / "without_inputs"
    without_inputs.mkdir()
    _write_config(without_inputs, with_inputs=False)
    with pytest.raises(AssertionError):
        L.assert_saved_scheme(str(without_inputs), "W8A8")


@pytest.mark.parametrize("weights_overrides", [
    {"strategy": "group", "group_size": 128},   # the stock W8A16 preset
    {"num_bits": 4},
    {"symmetric": False},
    {"type": "float"},
    {"group_size": 128},                        # channel label, group payload
])
def test_saved_scheme_rejects_weights_that_are_not_path_1s(
        tmp_path, weights_overrides):
    model_dir = tmp_path / "wrong_weights"
    model_dir.mkdir()
    _write_config(model_dir, weights_overrides, with_inputs=False)
    with pytest.raises(AssertionError):
        L.assert_saved_scheme(str(model_dir), PROBE)


def test_saved_scheme_accepts_both_channel_group_size_spellings(tmp_path):
    # The quantizer saves null; vLLM 0.29.0 needs -1 (see set_channel_group_size).
    for index, group_size in enumerate((None, -1)):
        model_dir = tmp_path / f"spelling{index}"
        model_dir.mkdir()
        _write_config(model_dir, {"group_size": group_size}, with_inputs=False)
        L.assert_saved_scheme(str(model_dir), PROBE)


# ---- vLLM 0.29.0 channel-wise WNA16 MoE workaround -----------------------
# The MoE WNA16 backend oracle compares the raw group_size (`group_size <= 0`)
# before normalising it, so a per-channel checkpoint saved with
# `group_size: null` raises TypeError while the engine initialises.

def _read_config(model_dir):
    return json.loads((model_dir / "config.json").read_text())


def test_set_channel_group_size_changes_only_that_field(tmp_path):
    _write_config(tmp_path, {"group_size": None}, with_inputs=False)
    before = _read_config(tmp_path)
    assert L.set_channel_group_size(str(tmp_path)) is True
    after = _read_config(tmp_path)
    assert after["quantization_config"]["config_groups"]["group_0"][
        "weights"]["group_size"] == -1
    after["quantization_config"]["config_groups"]["group_0"][
        "weights"]["group_size"] = None
    assert after == before


def test_set_channel_group_size_is_idempotent(tmp_path):
    _write_config(tmp_path, {"group_size": None}, with_inputs=False)
    assert L.set_channel_group_size(str(tmp_path)) is True
    written = (tmp_path / "config.json").read_bytes()
    assert L.set_channel_group_size(str(tmp_path)) is False
    assert (tmp_path / "config.json").read_bytes() == written
    assert not (tmp_path / "config.json.new").exists()


def test_set_channel_group_size_updates_all_groups(tmp_path):
    _write_path4_config(tmp_path)
    assert L.set_channel_group_size(str(tmp_path)) is True
    after = _read_config(tmp_path)["quantization_config"]["config_groups"]
    assert after["group_0"]["weights"]["group_size"] == -1
    assert after["group_1"]["weights"]["group_size"] == -1
    assert L.set_channel_group_size(str(tmp_path)) is False


def test_set_channel_group_size_refuses_group_wise_weights(tmp_path):
    _write_config(tmp_path, {"strategy": "group", "group_size": 128},
                  with_inputs=False)
    original = (tmp_path / "config.json").read_bytes()
    with pytest.raises(AssertionError):
        L.set_channel_group_size(str(tmp_path))
    assert (tmp_path / "config.json").read_bytes() == original


# ---- preflight geometry (no llmcompressor needed) -----------------------

def _lfm2_weight_map(attention_layers=None):
    """A synthetic safetensors weight map with LFM2.5-24B-A2B's layout."""
    attention_layers = (L.FULL_LAYERS if attention_layers is None
                        else attention_layers)
    weight_map = {}
    for layer in range(L.N_LAYERS):
        prefix = L.layer_prefix(layer)
        weight_map[f"{prefix}.operator_norm.weight"] = "shard.safetensors"
        if layer in attention_layers:
            weight_map[f"{prefix}.self_attn.q_proj.weight"] = "shard.safetensors"
        else:
            weight_map[f"{prefix}.conv.in_proj.weight"] = "shard.safetensors"
        if layer in L.DENSE_LAYER_IDS:
            weight_map[f"{prefix}.feed_forward.w1.weight"] = "shard.safetensors"
        else:
            weight_map[f"{prefix}.feed_forward.gate.weight"] = "shard.safetensors"
            for expert in range(L.N_EXPERTS):
                weight_map[f"{prefix}.feed_forward.experts.{expert}.w1.weight"] = (
                    "shard.safetensors")
    return weight_map


def _write_index(model_dir, weight_map):
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    return str(model_dir)


def test_preflight_accepts_the_real_lfm2_layout(tmp_path):
    # The CLI tests point --src at a missing directory (preflight skipped),
    # so a broken import inside preflight_geometry would otherwise only
    # surface on a real build.
    import quantize_lfm2_rtn
    info = quantize_lfm2_rtn.preflight_geometry(
        _write_index(tmp_path, _lfm2_weight_map()))
    assert info["layers"] == L.N_LAYERS
    assert info["moe_layers"] == L.N_MOE_LAYERS
    assert info["experts"] == L.N_EXPERTS


def test_preflight_rejects_a_wrong_attention_parity(tmp_path):
    import quantize_lfm2_rtn
    other_parity = list(range(3, 40, 4))
    with pytest.raises(SystemExit):
        quantize_lfm2_rtn.preflight_geometry(
            _write_index(tmp_path, _lfm2_weight_map(other_parity)))


# ---- recipe + CLI (need llmcompressor; run on the build node) --------------------

def _recipe_module():
    pytest.importorskip("llmcompressor")
    import quantize_lfm2_rtn
    return quantize_lfm2_rtn


def _resolved_group(recipe):
    return recipe.resolve_quantization_config().config_groups["group_0"]


def test_probe_recipe_keeps_path_1_weights_and_drops_activations():
    module = _recipe_module()
    shipped = _resolved_group(module.build_recipe("W8A8"))
    probe = _resolved_group(module.build_recipe(PROBE))
    assert probe.weights == shipped.weights
    assert shipped.input_activations is not None
    assert probe.input_activations is None
    assert probe.targets == shipped.targets == ["Linear"]


def test_probe_recipe_uses_the_same_ignore_list_as_path_1():
    module = _recipe_module()
    shipped = module.build_recipe("W8A8").resolve_quantization_config()
    probe = module.build_recipe(PROBE).resolve_quantization_config()
    assert list(probe.ignore) == list(shipped.ignore) == list(L.IGNORE_RTN)


def test_build_recipe_defaults_to_the_shipped_scheme():
    module = _recipe_module()
    assert (_resolved_group(module.build_recipe())
            == _resolved_group(module.build_recipe("W8A8")))


def test_build_recipe_rejects_an_unknown_scheme():
    module = _recipe_module()
    with pytest.raises(ValueError):
        module.build_recipe("W4A16")


def _run_dry(module, monkeypatch, capsys, tmp_path, *extra_args):
    argv = ["quantize_lfm2_rtn.py", "--dry-run",
            "--src", str(tmp_path / "no-such-source"), *extra_args]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return capsys.readouterr().out


def test_default_dry_run_is_the_unchanged_path_1_build(
        monkeypatch, capsys, tmp_path):
    module = _recipe_module()
    out = _run_dry(module, monkeypatch, capsys, tmp_path)
    assert f"dst:    {L.DEFAULT_DST}" in out
    assert "method: QuantizationModifier W8A8 RTN" in out
    assert "activations: INT8 dynamic per-token" in out


def test_probe_dry_run_targets_its_own_directory(
        monkeypatch, capsys, tmp_path):
    module = _recipe_module()
    out = _run_dry(module, monkeypatch, capsys, tmp_path, "--scheme", PROBE)
    assert f"dst:    {L.default_dst_for(PROBE)}" in out
    assert f"dst:    {L.DEFAULT_DST}\n" not in out
    assert f"method: QuantizationModifier {PROBE} RTN" in out
    assert "activations: BF16" in out


def test_probe_dry_run_refuses_the_shipped_directory(
        monkeypatch, capsys, tmp_path):
    module = _recipe_module()
    with pytest.raises(SystemExit):
        _run_dry(module, monkeypatch, capsys, tmp_path,
                 "--scheme", PROBE, "--dst", L.DEFAULT_DST)


def test_path4_recipe_splits_experts_w8a8_from_attn_dense_w8a16():
    module = _recipe_module()
    shipped = _resolved_group(module.build_recipe("W8A8"))
    cfg = module.build_recipe(PATH4).resolve_quantization_config()
    groups = cfg.config_groups
    assert set(groups) == {"group_0", "group_1"}
    g0, g1 = groups["group_0"], groups["group_1"]
    assert g0.weights == shipped.weights
    assert g1.weights == shipped.weights
    assert g0.input_activations == shipped.input_activations
    assert g1.input_activations is None
    assert list(g0.targets) == list(L.PATH4_W8A8)
    assert list(g1.targets) == list(L.PATH4_W8A16)
    assert list(cfg.ignore) == list(L.IGNORE_RTN)


def test_path4_dry_run_targets_its_own_directory_and_prints_split(
        monkeypatch, capsys, tmp_path):
    module = _recipe_module()
    out = _run_dry(module, monkeypatch, capsys, tmp_path, "--scheme", PATH4)
    assert f"dst:    {L.default_dst_for(PATH4)}" in out
    assert f"method: QuantizationModifier {PATH4} RTN" in out
    assert "7296 experts W8A8 + 46 attn/dense W8A16" in out
    with pytest.raises(SystemExit):
        _run_dry(module, monkeypatch, capsys, tmp_path,
                 "--scheme", PATH4, "--dst", L.DEFAULT_DST)


def test_census_checks_activation_mode_per_scheme():
    import quantize_lfm2_rtn
    import torch

    class _Scheme:
        def __init__(self, is_a8):
            self.input_activations = object() if is_a8 else None

    class _Toy(torch.nn.Module):
        def __init__(self, exp_a8=True, attn_a8=False, dense_a8=False):
            super().__init__()
            self.layers = torch.nn.ModuleDict({
                "0": torch.nn.ModuleDict({
                    "feed_forward": torch.nn.ModuleDict({
                        "w1": torch.nn.Linear(4, 4, bias=False),
                    }),
                }),
                "2": torch.nn.ModuleDict({
                    "self_attn": torch.nn.ModuleDict({
                        "q_proj": torch.nn.Linear(4, 4, bias=False),
                    }),
                    "feed_forward": torch.nn.ModuleDict({
                        "experts": torch.nn.ModuleDict({
                            "0": torch.nn.ModuleDict({
                                "w1": torch.nn.Linear(4, 4, bias=False),
                            }),
                        }),
                    }),
                }),
            })
            self.layers["2"]["feed_forward"]["experts"]["0"]["w1"].quantization_scheme = _Scheme(exp_a8)
            self.layers["2"]["self_attn"]["q_proj"].quantization_scheme = _Scheme(attn_a8)
            self.layers["0"]["feed_forward"]["w1"].quantization_scheme = _Scheme(dense_a8)

    c_ok = quantize_lfm2_rtn.census(_Toy(exp_a8=True, attn_a8=False, dense_a8=False), scheme=PATH4)
    assert c_ok["experts"] == 1 and c_ok["attn"] == 1 and c_ok["dense"] == 1
    assert c_ok["mismatch"] == []

    c_bad = quantize_lfm2_rtn.census(_Toy(exp_a8=True, attn_a8=True, dense_a8=False), scheme=PATH4)
    assert len(c_bad["mismatch"]) == 1 and "self_attn.q_proj" in c_bad["mismatch"][0]


