# LFM2.5-24B-A2B W8A8 — recipe

Four paths, same philosophy: **uniform RTN** (no GPTQ — no calib-distribution
favorites), router always BF16, short-conv block always BF16, leftovers BF16.

## Path 1 — RTN (daily driver)

`scripts/quantize_lfm2_rtn.py` — data-free `QuantizationModifier(scheme="W8A8",
targets=["Linear"])` with `IGNORE_RTN` from `scripts/lfm2_common.py`.

Ignore table:

```python
IGNORE_RTN = [
    r"re:.*lm_head",                    # no-op: tied embeddings store no head
    r"re:.*\.gate$",                    # MoE router = feed_forward.gate
    r"re:.*\.(?:conv|short_conv)\..*",  # whole short-conv block
    r"re:.*embed_tokens$",
]
```

Every expert (`experts.{e}.w1/w3/w2`), every attention projection, and the
two dense MLPs are quantized. Norms, `q/k_layernorm`, and `expert_bias` are
not `nn.Linear` and are skipped automatically.

Expected census: **7342** = 64×38×3 experts + 10×4 attention + 2×3 dense.

## Path 2 — SmoothQuant + RTN-E12 (extra accuracy, more VRAM saved)

> **Status (2026-10-17):** ran end to end; the smoothed-uniform build
> regressed (KLD 0.70 vs 0.22) because fold A left the router unscaled. Fixed,
> not rebuilt. The scale search is still defective. Evidence and next steps:
> `docs/accuracy-suggestions.md` § "Path 2 post-mortem".
>
> **Update (2026-10-18):** Path 3 (the W8A16-CHANNEL probe) shows the whole gap
> between Path 1 and Liquid's FP8 is activation quantization (cost 0.079 KLD /
> 4.3 pt top-1; INT8 weights alone beat FP8). Smoothing has to recover about
> 34-43% of that cost to match FP8. See Path 3 below and
> `docs/accuracy-suggestions.md` § "Path 3 — W8A16-CHANNEL probe".

`capture-lfm2.py` → `lfm2_alpha.py` → `apply-smoothing-lfm2.py` →
`quantize_lfm2_sq.py`.

- **Capture** (58 hooks: 1 per MoE layer + 2 per attention layer):
  routed-expert input (`{lk}.feed_forward` = post-`ffn_norm`, shared by the
  router and all 64 experts, A fold), plus attn `q_proj`/`out_proj` inputs on
  the 10 attention layers. There are no per-expert `w2` hooks (transformers
  fuses the experts on load); alpha recomputes each expert's down input from
  the routed-in rows. Conv layers get no hooks. Window selection is
  **strided**, not head-truncated.
- **Alpha**: base × alpha grid per class; `moe-in` pools consumer col-maxima
  over all 64 experts' `w1`+`w3`; `moe-down` picks one (base, alpha) per layer
  on the sampled experts, then gives each expert its own scale from its own
  amax; `attn-out` is GQA-constrained to `[N_KV=8, G=4, HD=64]`.
  **Known defect:** the grid has no identity candidate and uses true window
  maxima, which per-token dynamic quantization already absorbs. On held-out
  rows the chosen scales were worse than plain RTN for A (26/38 layers), C and
  E. Revision spec (measured on held-out windows: A +39%, B +66%, E +3%): add
  an identity candidate; use 99.9th-percentile stats of typical rows; exclude
  first-token rows; select on some windows and score on others; normalise `s`
  to unit geometric mean.
- **Smoothing folds** (all exact pre-quant; **norms are PLAIN**, `w' = w/s` —
  LFM2 is NOT zero-centered):
  - A moe-in: `{lk}.ffn_norm` (plain) → each expert's `w1` AND `w3` input
    cols × `s_a` AND the router's `feed_forward.gate` input cols × `s_a` (the
    router reads the same normed tensor; leave it out and it sees `x / s_a`)
  - C moe-down: per-expert `s_c[e]` → `w3` **rows** ÷ `s_c[e]`, `w2` cols ×
    `s_c[e]`. Up side only — the gate side must not be scaled (SiLU).
  - B attn-in: `{lk}.operator_norm` (plain) → q/k/v input cols × `s_b`. All
    three are required: `s_b` is per-channel, so the per-head q/k RMSNorms do
    not cancel it (q/k left unfolded → 21% attention-block error; as coded →
    4e-7).
  - E attn-out: `v_proj` rows ÷ `t`, `out_proj` cols × `t`, GQA-constrained.
- **E12 applier**: RTN INT8 with scheme groups — expert `w1`/`w3` + attn +
  dense `w1`/`w3` go W8A8; expert `w2` + dense `w2` stay W8A16. Plus a
  post-oneshot census proving the split (4908 / 2434).

  Expert tensors carry **three different names** depending on pipeline stage,
  and the E12 patterns cover all of them:

  | Stage | Name |
  |---|---|
  | on disk | `experts.{e}.w1` / `.w3` / `.w2` |
  | after HF load | `experts.gate_up_proj` / `experts.down_proj` (fused 3D) |
  | after llm-compressor linearization | `experts.{e}.gate_proj` / `.up_proj` / `.down_proj` |

  The third row is what the applier actually sees, because transformers fuses
  the 2D checkpoint on load and llm-compressor's `LinearExperts2D` then
  unfuses it into `gate_proj`/`up_proj`/`down_proj`. Missing that form leaves
  96% of the model at BF16 — the census reports `fused_experts` and fails
  loudly if linearization never ran.

  Note the router ignore `re:.*\.gate$` deliberately does **not** match
  `experts.{e}.gate_proj`, which is what keeps the 2432 expert gates
  quantizable in Path 1. There is a regression test for that collision.

### Open questions (status 2026-10-17)

1. **RESOLVED — the B fold and q/k layernorms.** Fold q, k AND v. The folds
   cancel exactly on the input side (4e-7 through HF's `Lfm2MoeAttention`,
   fp32). The earlier options (a) "v-only" and (b) are moot: v-only is 21% off
   because `s_b` is per-channel, not a scalar, so RMSNorm's scale invariance
   does not apply. Pinned by `tests/test_fold_plan.py`.
2. **Dense layers 0-1 are not smoothed** (no expert table, ~0.6% of params).
   They get plain RTN. Add a dense-down hook if that turns out to matter.
3. **OBSOLETE — experts that never route.** There are no per-expert `w2`
   hooks, so there is no empty-window case: every expert's down input is
   recomputed from the shared routed-in rows and gets its own scale. The
   stats therefore cover all sampled rows, not only the tokens routed to that
   expert.
4. **RESOLVED — fold parity test.** On real layer-3 weights, A+C with the
   router folded matches the source to 3.3e-7 in fp32 (BF16 storage: ~0.2%
   block error with routing forced, 0.5-1% with free routing from near-tie
   router flips). `tests/test_fold_plan.py` pins the wiring.

## Path 3 — Weight-only isolation probe (`--scheme W8A16-CHANNEL`)

`scripts/quantize_lfm2_rtn.py --scheme W8A16-CHANNEL` — data-free
`QuantizationModifier` with `config_groups={"group_0": QuantizationScheme(
targets=["Linear"], weights=PRESET_SCHEMES["W8A8"]["weights"])}` and the same
`IGNORE_RTN` keep-outs as Path 1.

- **Purpose:** isolates weight rounding from activation quantization and serves
  as our INT8 weight-only baseline on Ampere (where Liquid's FP8 also runs
  weight-only (BF16 activations, no INT8 compute path).
- **Why not stock `W8A16`:** in `compressed-tensors` 0.18 the stock `W8A16`
  preset is `group_size=128`, finer than Path 1's per-channel weights. Reusing
  Path 1's exact weight args guarantees the 7342 quantized weights are
  bit-identical to Path 1's.
- **vLLM 0.29.0 MoE quirk:** `set_channel_group_size()` rewrites
  `group_size: null` -> `-1` in the saved `config.json` so vLLM's
  `check_moe_marlin_supports_config` does not fail with `TypeError` on `None <= 0`.
- **Expected census:** **7342** Linears at W8A16 per-channel (same modules as
  Path 1). Measured (2026-10-18): KLD **0.137**, top-1 **84.1%**, PPL **29.04**,
  decode **~+1% vs FP8** (~3–4% faster TTFT on shared arXiv extracts).

## Path 4 — Selective component split: experts W8A8 + attention/dense W8A16 (`--scheme W8A8-EXPERTS`)

`scripts/quantize_lfm2_rtn.py --scheme W8A8-EXPERTS` — data-free two-group
`QuantizationModifier` with `IGNORE_RTN` keep-outs, using `PATH4_W8A8` and
`PATH4_W8A16` from `scripts/lfm2_common.py`:

- **`group_0` (W8A8, `int-quantized`):** all **7296** MoE expert Linears
  (`w1`, `w3`, `w2` across layers 2-39, matched across on-disk `w1/w3/w2`,
  linearized `gate_proj/up_proj/down_proj`, and fused `gate_up_proj/down_proj`
  names) with Path 1's per-channel INT8 weights + dynamic per-token INT8
  activations.
- **`group_1` (W8A16, `pack-quantized`):** all **40** attention Linears
  (`q/k/v/out_proj` on layers 2, 6, …, 38) + **6** dense MLP Linears (`w1/w3/w2`
  on layers 0-1) with Path 1's per-channel INT8 weights and
  `input_activations=None` (BF16 activations).
- **Reasoning:** Path 3 proved INT8 per-channel weights beat FP8 by 0.045 KLD /
  2.9 pt top-1, and that the entire Path 1 -> FP8 gap is activation
  quantization. MoE experts account for **95.7%** of parameters (22.95B) and the
  bulk of decode compute, whereas attention (0.4%) and dense layers 0-1 (0.6%)
  sit on every token's critical path (layers 0-1 condition the residual stream
  before any MoE layer runs). Keeping attention + dense 0-1 at W8A16 preserves
  BF16 activations there while putting all 38 MoE expert blocks on the W8A8
  INT8 tensor-core fast-path.
- **Why vLLM 0.29.0 serves this (unlike E12):** vLLM's
  `CompressedTensorsMoEMethod.get_moe_method` checks that `.0.gate_proj`,
  `.0.up_proj`, and `.0.down_proj` resolve to the same scheme dict. In Path 4
  all three are in the experts group (W8A8 -> method
  `CompressedTensorsW8A8Int8MoEMethod`, executed by the Triton Int8 MoE kernel
  `TritonExperts`), and every stacked dense linear
  (`qkv_proj` from `q/k/v_proj`, `w13` from `.feed_forward.w1/w3`) has all of
  its shards in the weight-only group (W8A16 -> method
  `CompressedTensorsWNA16`, executed by the AllSpark linear kernel).
  `set_channel_group_size()` writes `group_size: -1` across both groups. Note
  that `ModelCompressor` numbers `config_groups` in module-walk order, so on
  disk `group_0` is the `PATH4_W8A16` group (`layers.0` comes first) and
  `group_1` is the `PATH4_W8A8` experts group (`layers.2` comes later);
  `assert_saved_scheme` matches groups by their target sets.
- **Expected census:** **7342** total = **7296** experts (W8A8) + **40** attn
  (W8A16) + **6** dense (W8A16), `conv=0`, `router=0`, `mismatch=[]`.
  Measured (2026-10-18): KLD **0.153**, top-1 **82.9%**, PPL **28.88**, decode
  **+8.2..8.6% vs FP8** (200.4 / 189.9 / 187.8 / 176.9 tok/s across the 4 arXiv
  URLs; TTFT 661ms / 2.92s / 2.80s / 6.32s).
- **Execution:**
  1. Stop any active serve on the build node (`docker ps` empty on `:8051`).
  2. Dry-run: `../.venv/bin/python scripts/quantize_lfm2_rtn.py --scheme W8A8-EXPERTS --dry-run`
  3. Build (~10 min): `../.venv/bin/python -u scripts/quantize_lfm2_rtn.py --scheme W8A8-EXPERTS > logs/path4-w8a8-experts.log 2>&1`
  4. Structural check: `../.venv/bin/python scripts/smoke.py --model ~/models/LFM2.5-24B-A2B-W8A8-EXPERTS --expect-experts 7296`
  5. Serve via `lfm-w8a8/eval/composes/docker-compose.lfm-w8a8-experts.yml`
     (identical flags to FP8/W8A8/W8A16), warm up from the client host, run the 4-URL
     extract speed leg (`testing/lfm-w8a8-experts/`) and the 240-window KLD
     capture (`kldfv-lfm-w8a8-experts.npz`) + floor-free paired comparison.

## Geometry (verified against config + safetensors at runtime by
## `quantize_lfm2_rtn.preflight_geometry`)

40 layers — attention at idx 2, 6, …, 38 (10) and gated short conv at the
other 30; hidden 2048; 32 Q / 8 KV / head_dim 64; 64 routed experts, top-4,
`moe_intermediate` 1536, **no shared expert**; dense MLP on layers 0-1
(inter 11776); 128K vocab, tied embeddings; experts stored **unfused** as
per-expert 2D `nn.Linear`s (`w1`/`w3`/`w2`) with `.weight` suffixes.

## References & Heritage

- **[`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8):**
  source of the teacher-forced full-vocab KLD + floor-free paired evaluation
  harness (`kld-fullvocab.py`, `kld-paired-floorfree.py`), the mixed-precision
  `compressed-tensors` `config_groups` architecture (`int-quantized` W8A8 +
  `pack-quantized` W8A16), the component-level activation-error study, the
  E12 (`down_proj` W8A16) hypothesis, and the offline SmoothQuant proxy design.
- **[`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)**
  & **[`LiquidAI/LFM2.5-24B-A2B-FP8`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-FP8):**
  source BF16 weights and official FP8 reference / keep-out list (`IGNORE_RTN`).
