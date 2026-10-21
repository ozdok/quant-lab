"""Shared geometry + target/ignore tables for LFM2.5-24B-A2B (LFM2.5 MoE).

Architecture (verified against config.json + the safetensors headers on the
published checkpoint, not against documentation):
  40 layers, hidden 2048, 32 Q heads / 8 KV heads / head_dim 64, 128K vocab,
  tied embeddings, no MTP tower, no vision tower.
  Full-attention on idx 2, 6, ..., 38 (10 layers). The other 30 are gated
  short convolutions (``Lfm2MoeShortConv``) — NOT linear attention (no
  recurrent state math; the non-attention layers here are convolutions).
  Layers 0-1 carry a dense MLP (inter 11776); layers 2-39 carry the MoE
  block (64 routed experts, top-4, moe_intermediate 1536). No shared expert.

Source-checkpoint reality — the wiring here is the OPPOSITE of the fused
MoE layout::

  Fused (other MoE families): mlp.experts.gate_up_proj  FUSED nn.Parameter,
                      no .weight suffix
  LFM2 (this model):  feed_forward.experts.{E}.w1.weight [1536,2048]
                      feed_forward.experts.{E}.w3.weight [1536,2048]
                      feed_forward.experts.{E}.w2.weight [2048,1536]
                      per-expert 2D nn.Linear, WITH .weight suffix

Because the experts ship already-unfused as ordinary 2D Linears,
``targets=["Linear"]`` catches all 7296 of them directly and there is no
fused-half bookkeeping anywhere. That also removes the structural cause of
the fused-halves C-fold bug (docs/accuracy-suggestions.md item 1): gate
and up are separate tensors here, so the moe-down fold scales ``w3`` rows
only.

Module-name map that the ignore tables depend on (all verified on disk):
  model.embed_tokens.weight                       nn.Embedding  (tied)
  model.layers.N.conv.conv.weight                 Conv1d depthwise
  model.layers.N.conv.in_proj.weight              Linear [6144, 2048]
  model.layers.N.conv.out_proj.weight             Linear [2048, 2048]
  model.layers.N.self_attn.{q,k,v,out}_proj       Linear (attn layers)
  model.layers.N.self_attn.{q,k}_layernorm        RMSNorm [64]
  model.layers.N.feed_forward.gate.weight         Linear [64, 2048] ROUTER
  model.layers.N.feed_forward.experts.{E}.w{1,2,3} Linear
  model.layers.N.feed_forward.w{1,2,3}            Linear (dense, idx < 2)
  model.layers.N.{operator_norm,ffn_norm}         RMSNorm [2048]
There is no ``mlp.`` anywhere and no ``linear_attn.`` anywhere.

No heavy imports — safe for unit tests.
"""

import os
import re

# ---- Geometry (verified against the published checkpoint config) ----
N_LAYERS = 40
HIDDEN = 2048
N_Q, N_KV, HD = 32, 8, 64
G = N_Q // N_KV  # 4 q-heads per kv group

N_EXPERTS = 64
TOPK = 4
MOE_INTER = 1536
# LFM2.5-24B-A2B has NO shared expert.
N_SHARED = 0

# First `DENSE_LAYERS` layers use a dense MLP instead of the MoE block.
DENSE_LAYERS = 2
DENSE_INTER = 11776
N_MOE_LAYERS = N_LAYERS - DENSE_LAYERS  # 38

# Gated short convolution (Lfm2MoeShortConv) geometry.
CONV_KERNEL = 3  # config.conv_L_cache
CONV_IN_PROJ = 3 * HIDDEN  # [6144, 2048]

# Full-attention layers: every 4th starting at 2 → 2, 6, ..., 38.
FULL_LAYERS = list(range(2, N_LAYERS, 4))
N_ATTN = len(FULL_LAYERS)
N_CONV = N_LAYERS - N_ATTN
assert N_ATTN == 10 and N_CONV == 30, (N_ATTN, N_CONV)

# Exact layer-index sets. The parity matters: attention on 3, 7, ..., 39
# would also give 10 attention + 30 non-attention layers, so any geometry
# check must compare indices, never just counts.
CONV_LAYER_IDS = [i for i in range(N_LAYERS) if i not in FULL_LAYERS]
DENSE_LAYER_IDS = list(range(DENSE_LAYERS))
MOE_LAYER_IDS = list(range(DENSE_LAYERS, N_LAYERS))
assert len(CONV_LAYER_IDS) == N_CONV
assert len(MOE_LAYER_IDS) == N_MOE_LAYERS

VOCAB = 128000
# tie_word_embeddings=true, so lm_head is NOT stored on disk.
TIED_EMBEDDINGS = True

# Router geometry (feed_forward.gate), kept BF16.
ROUTER_SHAPE = (N_EXPERTS, HIDDEN)
# Per-layer expert bias buffer, F32, not a Linear — untouched.
HAS_EXPERT_BIAS = True

# ---- Source tensor layout ----
# Per-expert 2D Linears ON DISK. (Not fused — see module docstring.)
EXPERT_WEIGHT_SUFFIXES = ("w1", "w3", "w2")  # gate, up, down
EXPERT_GATE, EXPERT_UP, EXPERT_DOWN = "w1", "w3", "w2"

# ---- Entry-point defaults ----
DEFAULT_MODEL_NAME = "LFM2.5-24B-A2B"


def layer_prefix(i: int) -> str:
    """Full module path prefix for layer ``i``.

    LFM2 has no ``language_model`` wrapper — this is the single most common
    line to get wrong when adapting module paths from another family.
    """
    return f"model.layers.{i}"


def is_full_attn_layer(i: int) -> bool:
    """True when layer ``i`` is full attention rather than short conv."""
    return i in FULL_LAYERS


def is_moe_layer(i: int) -> bool:
    """True when layer ``i`` carries the MoE block rather than a dense MLP."""
    return i >= DENSE_LAYERS


# ---- RTN W8A8 recipe (Path 1) ----
# Data-free QuantizationModifier RTN, targets Linear, ignores below.
# Mirrors Liquid AI's own published FP8 ignore list for this model (the
# closest thing to an authored recipe that exists for LFM2.5-24B-A2B).
IGNORE_RTN = [
    r"re:.*lm_head",                    # no-op: tied embeddings store no head
    # MoE router. LFM2 names it `feed_forward.gate`, NOT `mlp.gate` — the
    # `mlp.gate` pattern silently misses it and quantizes the router, which
    # both AGENTS.md and llm-compressor's own MoE example forbid.
    r"re:.*\.gate$",
    # Entire short-conv block. `conv.conv` is a depthwise Conv1d (immune to
    # targets=["Linear"] anyway) but `conv.in_proj` [6144,2048] and
    # `conv.out_proj` [2048,2048] ARE Linears and would be caught. Liquid
    # keeps the whole block at source precision; follow them.
    r"re:.*\.(?:conv|short_conv)\..*",
    r"re:.*embed_tokens$",
]

# ---- SmoothQuant + RTN-E12 variant (Path 2) ----
# Builds and passes the census, but vLLM 0.29.0 cannot serve mixed-scheme MoE.
# Splits experts by activation sensitivity: gate/up inputs are post-norm and
# well behaved, so they go W8A8; the down-proj input is a gated product
# (silu(gate)*up) with spiky, position-dynamic error, so it is held at W8A16
# for weights+BF16 activations. Mirrors the Freaksterz E12 finding.
#
# Patterns are name-shape-agnostic on purpose. Expert tensors appear in up to
# three naming forms depending on where in the load/expand pipeline we are:
#   on disk          : experts.{e}.w1 / w3 / w2   (per-expert 2D nn.Linear)
#   after HF load    : experts.gate_up_proj / down_proj   (fused 3D)
#   after llm-comp   : experts.{e}.gate_proj / up_proj / down_proj
#                      (LinearExperts2D — verified in the installed
#                       llmcompressor/modeling/moe/linear_experts.py)
# All three must be covered or experts silently escape quantization.
E12_W8A8 = [
    r"re:.*[.]self_attn[.]q_proj$",
    r"re:.*[.]self_attn[.]k_proj$",
    r"re:.*[.]self_attn[.]v_proj$",
    r"re:.*[.]self_attn[.]out_proj$",
    r"re:.*[.]experts[.].*w1$",          # on-disk gate
    r"re:.*[.]experts[.].*w3$",          # on-disk up
    r"re:.*[.]experts[.].*gate_proj$",   # linearized gate
    r"re:.*[.]experts[.].*up_proj$",     # linearized up
    r"re:.*[.]experts[.].*gate_up_proj$",  # fused gate+up
    r"re:.*[.]feed_forward[.]w1$",       # dense layers 0-1, gate
    r"re:.*[.]feed_forward[.]w3$",       # dense layers 0-1, up
]
E12_W8A16 = [
    r"re:.*[.]experts[.].*w2$",          # on-disk down
    r"re:.*[.]experts[.].*down_proj$",   # linearized / fused down
    r"re:.*[.]feed_forward[.]w2$",       # dense layers 0-1, down
]

# E12 ignore: same keep-outs as RTN (router, conv block, embeddings, head).
IGNORE_SQ = list(IGNORE_RTN)

# ---- Path 4: experts-only W8A8 + attention/dense W8A16 (`W8A8-EXPERTS`) ----
# Keeps each vLLM fused unit uniform inside itself so vLLM 0.29.0 serves it:
#   group_0 W8A8  : all 7296 MoE expert Linears (w1, w3, w2 across layers 2-39)
#   group_1 W8A16 : all 40 attention Linears (q/k/v/out across 10 layers) and
#                   all 6 dense MLP Linears (w1/w3/w2 across layers 0-1)
# Both groups use Path 1's exact per-channel INT8 weight args; only
# input_activations differs.
PATH4_W8A8 = [
    r"re:.*[.]experts[.].*w1$",            # on-disk gate
    r"re:.*[.]experts[.].*w3$",            # on-disk up
    r"re:.*[.]experts[.].*w2$",            # on-disk down
    r"re:.*[.]experts[.].*gate_proj$",     # linearized gate
    r"re:.*[.]experts[.].*up_proj$",       # linearized up
    r"re:.*[.]experts[.].*down_proj$",     # linearized / fused down
    r"re:.*[.]experts[.].*gate_up_proj$",  # fused gate+up
]
PATH4_W8A16 = [
    r"re:.*[.]self_attn[.]q_proj$",
    r"re:.*[.]self_attn[.]k_proj$",
    r"re:.*[.]self_attn[.]v_proj$",
    r"re:.*[.]self_attn[.]out_proj$",
    r"re:.*[.]feed_forward[.]w1$",         # dense layers 0-1, gate
    r"re:.*[.]feed_forward[.]w3$",         # dense layers 0-1, up
    r"re:.*[.]feed_forward[.]w2$",         # dense layers 0-1, down
]

# ---- Post-oneshot census ----
# Expected counts of quantized Linear modules. Printed by the appliers and
# asserted by the tests; a mismatch means the build must not be trusted.
CENSUS_W8A8 = {
    "experts.gate": N_EXPERTS * N_MOE_LAYERS,   # w1
    "experts.up": N_EXPERTS * N_MOE_LAYERS,     # w3
    "attn.qkvo": N_ATTN * 4,
    "dense.gate_up": DENSE_LAYERS * 2,          # w1 + w3
}
CENSUS_W8A16 = {
    "experts.down": N_EXPERTS * N_MOE_LAYERS,   # w2
    "dense.down": DENSE_LAYERS,                 # w2
}
CENSUS_PATH4_W8A8 = {
    "experts": N_EXPERTS * N_MOE_LAYERS * 3,    # 7296
}
CENSUS_PATH4_W8A16 = {
    "attn": N_ATTN * 4,                         # 40
    "dense": DENSE_LAYERS * 3,                  # 6
}
# Path 1 (uniform W8A8) totals, for the RTN census.
CENSUS_RTN_TOTAL = (
    N_EXPERTS * N_MOE_LAYERS * 3 + N_ATTN * 4 + DENSE_LAYERS * 3
)

# Alpha-proxy search grid (same base x alpha grid as Freaksterz's alpha-proxy).
ALPHAS = [0.5, 0.65, 0.8, 0.9]
S_MIN, S_MAX = 1e-5, 16.0


def regex_for(suffix: str) -> str:
    """'feed_forward.experts.down_proj' -> 're:.*\\.feed_forward\\.experts\\.down_proj$'."""
    return "re:.*\\." + suffix.replace(".", "\\.") + "$"


def matches(name: str, pattern: str) -> bool:
    """compressed-tensors target semantics: 're:' -> regex fullmatch."""
    if pattern.startswith("re:"):
        return re.fullmatch(pattern[3:], name) is not None
    return name == pattern or name.endswith("." + pattern)


def pop_hook_samples(samples: dict, hook: str):
    """Concatenate one hook's sample shards and release them.

    Callers that need every hook exactly once should pop-on-use instead of
    cat-ing the whole dict up front: the full concat briefly holds TWO
    copies of all samples and can OOM a 64GB host, while popping peaks near
    one copy. Numerics are identical — same shards, same order, same
    concatenation.
    """
    import torch  # local import: keeps this module safe for light unit tests

    parts = samples.pop(hook, None)
    assert parts, f"no samples for {hook}"
    return torch.cat(parts).float()


def models_dir() -> str:
    """Workspace model root on whichever host runs the script."""
    env = os.environ.get("LFM2_MODELS")
    if env:
        return env
    return os.path.expanduser("~/models")


MODELS_DIR = models_dir()

DEFAULT_SRC_CANDIDATES = [
    DEFAULT_MODEL_NAME,
    f"{DEFAULT_MODEL_NAME}-Base",
]

DEFAULT_SRC = os.path.join(MODELS_DIR, DEFAULT_MODEL_NAME)
DEFAULT_DST = os.path.join(MODELS_DIR, f"{DEFAULT_MODEL_NAME}-W8A8")


# ---- RTN scheme selection (Path 1 W8A8, Path 3 W8A16-CHANNEL, Path 4 W8A8-EXPERTS) ----
# Path 3 answers "how much of Path 1's KLD is activation quantization?" by
# removing ONLY the activation quantizer. compressed-tensors 0.18's stock
# "W8A16" preset is group-128 weights, finer than the per-channel weights
# Path 1 ships, so it would flatter the weight error; Path 3 therefore
# reuses Path 1's exact weight args under its own name.
# Path 4 ("W8A8-EXPERTS") puts all 7296 MoE expert Linears at W8A8 while
# keeping the 40 attention Linears and 6 dense MLP Linears (layers 0-1) at
# per-channel W8A16.
RTN_SCHEMES = ("W8A8", "W8A16-CHANNEL", "W8A8-EXPERTS")


def require_known_scheme(scheme: str) -> None:
    if scheme not in RTN_SCHEMES:
        raise ValueError(
            f"unknown scheme {scheme!r}; choose from {RTN_SCHEMES}")


def default_dst_for(scheme: str) -> str:
    """Default output directory; W8A8 stays the shipped Path 1 directory."""
    require_known_scheme(scheme)
    return os.path.join(MODELS_DIR, f"{DEFAULT_MODEL_NAME}-{scheme}")


def assert_not_shipped_dst(scheme: str, dst: str) -> None:
    """Refuse to let a non-shipped scheme write into the shipped W8A8 dir.

    ``--dst`` defaults to that directory and oneshot() writes into it in
    place, so a probe build with a forgotten ``--dst`` would silently replace
    the daily-driver checkpoint. Path 1 itself may still rebuild in place.
    """
    require_known_scheme(scheme)
    if scheme == "W8A8":
        return
    if os.path.realpath(dst) == os.path.realpath(DEFAULT_DST):
        raise SystemExit(
            f"refusing: --scheme {scheme} would write into the shipped "
            f"Path 1 directory {DEFAULT_DST}. Pass a different --dst.")


def _assert_path1_weights(weights: dict, label: str = "group_0") -> None:
    expected = {"num_bits": 8, "type": "int", "symmetric": True,
                "strategy": "channel"}
    wrong = {k: (weights.get(k), v) for k, v in expected.items()
             if weights.get(k) != v}
    # Channel-wise weights carry no group size: null as saved by the
    # quantizer, or -1 once set_channel_group_size has run.
    if weights.get("group_size") not in (None, -1):
        wrong["group_size"] = (weights.get("group_size"), "None or -1")
    assert not wrong, (
        f"{label}: saved weights are not Path 1's per-channel int8 RTN "
        f"(saved, expected): {wrong}")


def assert_saved_scheme(model_dir: str, scheme: str) -> None:
    """Fail loudly if a saved checkpoint is not the requested scheme.

    All schemes must carry Path 1's weights (int8, symmetric, per-channel);
    they differ only in which groups carry an input-activation quantizer.
    """
    import json
    require_known_scheme(scheme)
    with open(os.path.join(model_dir, "config.json")) as fh:
        groups = json.load(fh)["quantization_config"]["config_groups"]
    if scheme == "W8A8-EXPERTS":
        assert set(groups) == {"group_0", "group_1"}, (
            f"scheme {scheme}: expected groups {{'group_0', 'group_1'}}, "
            f"got {set(groups)}")
        for gname, grp in groups.items():
            _assert_path1_weights(grp["weights"], gname)
        by_targets = {tuple(grp.get("targets", ())): grp
                      for grp in groups.values()}
        exp_grp = by_targets.get(tuple(PATH4_W8A8))
        a16_grp = by_targets.get(tuple(PATH4_W8A16))
        assert exp_grp is not None and a16_grp is not None, (
            f"scheme {scheme}: expected target sets PATH4_W8A8 and PATH4_W8A16, "
            f"got {[grp.get('targets') for grp in groups.values()]}")
        assert exp_grp.get("input_activations") is not None, (
            f"scheme {scheme}: experts group missing input_activations")
        assert a16_grp.get("input_activations") is None, (
            f"scheme {scheme}: attn+dense group must have no input_activations")
        return

    assert set(groups) == {"group_0"}, (
        f"scheme {scheme}: expected single group_0, got {set(groups)}")
    group = groups["group_0"]
    _assert_path1_weights(group["weights"], "group_0")
    has_inputs = group.get("input_activations") is not None
    assert has_inputs == (scheme == "W8A8"), (
        f"scheme {scheme}: input_activations "
        f"{'present' if has_inputs else 'absent'} in the saved config")


def set_channel_group_size(model_dir: str) -> bool:
    """Write ``group_size: -1`` for channel-wise weights in config.json.

    compressed-tensors treats -1 as the channel-wise convention, but the
    quantizer saves ``null``. vLLM 0.29.0's WNA16 MoE backend oracle compares
    the raw value (``group_size <= 0``) before normalising it, so a
    per-channel weight-only checkpoint raises TypeError while the engine
    initialises. Updates every config_group in place; weights are untouched.
    Returns True when the file changed.
    """
    import json
    path = os.path.join(model_dir, "config.json")
    with open(path) as fh:
        config = json.load(fh)
    groups = config["quantization_config"]["config_groups"]
    changed = False
    for gname, group in groups.items():
        weights = group["weights"]
        assert weights["strategy"] == "channel", (
            f"{gname}: refusing to set group_size on {weights['strategy']!r} weights")
        if weights.get("group_size") == -1:
            continue
        assert weights.get("group_size") is None, (
            f"{gname}: unexpected channel-wise group_size {weights['group_size']!r}")
        weights["group_size"] = -1
        changed = True
    if not changed:
        return False
    tmp_path = path + ".new"
    with open(tmp_path, "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_path, path)
    return True


# ---- Post-save expert-name restore (Path 1 ship-blocker, 2026-10-14) ----
# llmcompressor linearizes already-unfused LFM2 experts (2D -> 3D -> 2D) and
# saves them as gate_proj / up_proj / down_proj. Transformers and vLLM expect
# the original w1 (gate) / w3 (up) / w2 (down). Serving without this rename
# leaves 7296 experts unmapped -> fluent gibberish (single-token repeat).
# Shapes are 1:1 ([1536,2048]), only names change; data_offsets untouched.
EXPERT_NAME_FIXUPS = (
    ('.gate_proj.', '.w1.'),
    ('.up_proj.', '.w3.'),
    ('.down_proj.', '.w2.'),
)

def fixed_expert_name(k: str) -> str:
    if '.feed_forward.experts.' in k:
        for a, b in EXPERT_NAME_FIXUPS:
            k = k.replace(a, b)
    return k

def restore_expert_names(model_dir: str) -> int:
    """Rename linearized expert tensors back to w1/w3/w2 in-place.
    Rewrites safetensors headers + index.json. Returns renamed count."""
    import glob, json, shutil, struct
    ip = os.path.join(model_dir, 'model.safetensors.index.json')
    d = json.load(open(ip))
    wm = d['weight_map']
    new_wm = {fixed_expert_name(k): v for k, v in wm.items()}
    renamed_idx = sum(1 for a in wm if fixed_expert_name(a) != a)
    d['weight_map'] = new_wm
    json.dump(d, open(ip, 'w'), indent=2)
    total = 0
    for shard in sorted(glob.glob(os.path.join(model_dir, 'model-*.safetensors'))):
        with open(shard, 'rb') as f:
            (n,) = struct.unpack('<Q', f.read(8))
            h = json.loads(f.read(n))
            start = 8 + n
        names = [k for k in h if k != '__metadata__']
        need = sum(1 for k in names if fixed_expert_name(k) != k)
        if not need:
            continue
        new_h = {}
        for k, v in h.items():
            nk = fixed_expert_name(k) if k != '__metadata__' else k
            if nk in new_h:
                raise RuntimeError(f'name collision: {nk}')
            new_h[nk] = v
        nb = json.dumps(new_h).encode()
        tmp = shard + '.renamed'
        with open(shard, 'rb') as fin, open(tmp, 'wb') as fout:
            fin.seek(start)
            fout.write(struct.pack('<Q', len(nb)))
            fout.write(nb)
            shutil.copyfileobj(fin, fout, length=64*1024*1024)
        st = os.stat(shard)
        os.replace(tmp, shard)
        os.chmod(shard, st.st_mode)
        total += need
    assert total == renamed_idx, (total, renamed_idx)
    return total

def assert_saved_expert_names(model_dir: str) -> None:
    """Fail loudly if a saved checkpoint still carries linearized names."""
    import glob, json, struct
    bad = []
    for shard in sorted(glob.glob(os.path.join(model_dir, 'model-*.safetensors'))):
        with open(shard, 'rb') as f:
            (n,) = struct.unpack('<Q', f.read(8))
            h = json.loads(f.read(n))
        for k in h:
            if '.feed_forward.experts.' in k and ('gate_proj' in k or 'up_proj' in k or 'down_proj' in k):
                bad.append(k)
                break
    if bad:
        raise AssertionError(f'{len(bad)} shards still carry linearized expert names, e.g. {bad[0]}')
