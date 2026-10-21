# LFM2.5-24B-A2B W8A8

W8A8 INT8 (`compressed-tensors`) for **LFM2.5-24B-A2B** — Liquid AI's 24B MoE
reasoning model (2B active/token). No GPTQ anywhere (design choice: GPTQ
prioritizes by calibration activations and overfits to the corpus; RTN treats
every weight equally). SmoothQuant folds are exact math, not fitting, so
they're allowed.

Every module name below was verified against the published checkpoint's
safetensors headers and config — names that look interchangeable across model
families fail silently here (see AGENTS.md).

## Source geometry (verified from safetensors headers + config)

- 40 layers: 10 full-attention (idx **2, 6, …, 38**) + 30 **gated short
  convolution** layers. Not GDN — there is no `linear_attn` anywhere.
- Hidden 2048, 32 Q / 8 KV heads, head_dim 64 (G = 4 per KV group)
- 64 routed experts, top-4, `moe_intermediate` 1536 — **no shared expert**
- Layers 0-1 carry a dense MLP (inter 11776); layers 2-39 carry the MoE block
- Router uses **sigmoid** scores + an F32 `expert_bias` buffer, top-4
- 128K vocab, **tied** embeddings (no `lm_head` tensor on disk), 131K context
- Experts ship **UNFUSED** as per-expert 2D Linears:
  `feed_forward.experts.{e}.w1.weight` [1536,2048] (gate),
  `.w3.weight` [1536,2048] (up), `.w2.weight` [2048,1536] (down)
- Source: `LiquidAI/LFM2.5-24B-A2B` on HF, **47.94 GB** in 10 shards.
  It does **not** fit in the build node's RAM.

Parameter split (why this model is W8A8-friendly):

| Component | Params | Share |
|---|---|---|
| Expert FFNs (64 × 38 layers) | 22.95B | **95.7%** |
| Dense MLPs (layers 0-1) | 0.145B | 0.6% |
| Short-conv block (30 layers) | 0.504B | 2.1% |
| Attention (10 layers) | 0.105B | 0.4% |
| Embeddings (tied) | 0.262B | 1.1% |

## Four paths

### Path 1 — RTN W8A8 (daily driver, READY)

`scripts/quantize_lfm2_rtn.py` — data-free `QuantizationModifier(
scheme="W8A8", targets=["Linear"])` with `IGNORE_RTN` from
`scripts/lfm2_common.py`.

Because the experts are already per-expert 2D Linears **on disk**,
`targets=["Linear"]` catches all 7296 of them directly for RTN. Caveat
learned 2026-10-16: transformers **fuses** them into 3D params on load,
so capture-time per-expert hooks do not exist. C scales are recomputed
offline in exact BF16 instead (see build log).

Kept in BF16:

- `feed_forward.gate` (router) + `expert_bias`
- **Whole short-conv block** (`conv.*`, including its `in_proj`/`out_proj`
  Linears) — matches Liquid AI's own published FP8 ignore list for this model
- `embed_tokens`, all norms, `q_layernorm`/`k_layernorm`

Expected census: **7342** quantized Linears.

Status: **shipped 2026-10-15** — `LFM2.5-24B-A2B-W8A8` (24 GB, 2 shards,
census 7342, smoke OK, serving under vLLM). Measured vs BF16: KLD 0.216,
top-1 79.7%, PPL 29.32 -> 29.21, decode ~+7% vs FP8.

### Path 2 — SmoothQuant + RTN (ran 2026-10-16/17; NOT shipped)

`capture-lfm2.py` → `lfm2_alpha.py` → `apply-smoothing-lfm2.py` →
`quantize_lfm2_sq.py` (E12) or `quantize_lfm2_rtn.py` on the smoothed
checkpoint (uniform scheme, the only kind vLLM 0.29.0 can serve).

Status: the smoothed-uniform build (SQ-W8A8) regressed vs Path 1 (KLD 0.70 vs
0.216) because fold A left the router unscaled; fixed 2026-10-17, not rebuilt.
The scale search in `lfm2_alpha.py` is still defective. Read
`docs/accuracy-suggestions.md` § "Path 2 post-mortem" before rebuilding.

Two structural facts make the LFM2 folds exact:

- **The C fold is exact and per-expert.** LFM2's `w3` (up) and `w2` (down)
  are separate tensors, so the moe-down fold divides **only `w3` rows** and
  multiplies `w2` cols — which cancels exactly through `silu(gate) * up`.
  (Dividing both sides of a fused gate/up tensor would *not* be exact,
  because the gate side passes through SiLU.)
- **There is no shared expert**, and per-expert `w2`
  hooks do not exist after the HF load (transformers fuses the experts), so
  `lfm2_alpha.py` recomputes every expert's down input in exact BF16 from the
  shared routed-in rows and the stored `w1`/`w3`.

There is no GDN class at all (no `in_proj_qkv`/`in_proj_z`/`out_proj` folds),
and the short-conv block stays BF16 as in Path 1.

### Path 3 — Weight-only isolation probe / W8A16-CHANNEL (READY)

`scripts/quantize_lfm2_rtn.py --scheme W8A16-CHANNEL` — builds Path 1's exact
per-channel INT8 weights (all 7342 Linears, same `IGNORE_RTN`) with the input
activation quantizer removed (`input_activations=None`), plus automated
`group_size: -1` normalization in `config.json` so vLLM 0.29.0's Marlin MoE
oracle accepts per-channel W8A16. Not the stock `W8A16` preset (which is
group-128 and would flatter weight rounding).

Status: **built & verified 2026-10-18** — `LFM2.5-24B-A2B-W8A16-CHANNEL`
(24 GB, 2 shards, census 7342, sampled weights bit-identical to Path 1).
Measured vs BF16: **KLD 0.137, top-1 84.1%, PPL 29.04** — beats Liquid's FP8
(0.182 / 81.2% / 29.29) on every accuracy metric and matches FP8 speed (~+1%
decode, ~3–4% faster TTFT on shared arXiv extracts; ~6% below Path 1 W8A8's
INT8 compute fast-path). Isolates the entire Path 1 -> FP8 gap to activation
quantization (0.079 KLD / 4.3 pt top-1) and establishes our accuracy baseline
for selective W8A8/W8A16 component splits. Details:
`docs/accuracy-suggestions.md` § "Path 3 — W8A16-CHANNEL probe".

### Path 4 — Selective component split: experts W8A8 + attention/dense W8A16 (READY)

`scripts/quantize_lfm2_rtn.py --scheme W8A8-EXPERTS` — two-group data-free RTN
starting from Path 3 (`W8A16-CHANNEL`) and moving only the MoE experts onto the
W8A8 INT8 compute fast-path:

- **`group_0` / `group_1` split:** all **7296** MoE expert Linears
  (`w1`, `w3`, `w2` across layers 2-39 — `PATH4_W8A8`) at W8A8
  (`int-quantized`), and all **40** attention Linears (`q/k/v/out_proj` on
  layers 2, 6, …, 38) plus all **6** dense MLP Linears (`w1/w3/w2` on layers
  0-1 — `PATH4_W8A16`) at per-channel W8A16 (`pack-quantized`, `group_size: -1`).
- **Kept in BF16 (`IGNORE_RTN`):** router (`feed_forward.gate`), whole
  short-conv block (`conv.*`), `embed_tokens`, norms, `expert_bias`.

**Why this split works in vLLM 0.29.0 (where E12 failed):** E12 split `w1/w3`
(W8A8) from `w2` (W8A16) *inside* the same MoE expert block, tripping
`CompressedTensorsMoEMethod.get_moe_method`'s uniform-scheme check across
`.0.gate_proj`, `.0.up_proj`, `.0.down_proj`. Path 4 keeps every vLLM fused
unit uniform inside itself (`RoutedExperts` all W8A8 -> method
`CompressedTensorsW8A8Int8MoEMethod`, run by the Triton Int8 MoE kernel;
`qkv_proj`, `out_proj`, dense `w13` and
`w2` all W8A16 -> method `CompressedTensorsWNA16`, run by the AllSpark linear
kernel).

Status: **built & verified 2026-10-18** — `LFM2.5-24B-A2B-W8A8-EXPERTS`
(24 GB, 2 shards, census 7296 W8A8 + 46 W8A16, smoke OK). Measured vs BF16:
**KLD 0.153, top-1 82.9%, PPL 28.88** — beats Liquid's FP8 (0.182 / 81.2% /
29.29) AND achieves **+8.2% to +8.6% decode speedup over FP8** (200.4 / 189.9 /
187.8 / 176.9 tok/s across the 4 arXiv URLs), recovering **80% of Path 1's
activation quantization KLD cost** while keeping the full W8A8 INT8 tensor-core
speedup. Details: `docs/accuracy-suggestions.md` § "Path 4 — W8A8-EXPERTS
selective split".

## Benchmark summary (same harness across all builds)

Statistical leg: teacher-forced full-vocab KLD vs offline BF16 reference
(`transformers`), candidates served through vLLM `v0.29.0` (TP2 on 2× RTX 3090
Ti / 3090, Ampere SM86), 240 × 512-token WikiText-2 test windows (122,640
positions, top-K 512). Speed leg: sequential `extract_test.py` from `the client host` at
`temperature=0`, `--enable-thinking` across 4 arXiv HTML papers (5.5K, 20.9K,
24.1K, 50.6K prompt tokens).

| Checkpoint | Scheme | Mean KLD vs BF16 | Paired Δ vs FP8 (floor-free) | Top-1 agree | PPL (BF16 29.32) | Decode tok/s (5.5K / 20.9K / 24.1K / 50.6K) |
|---|---|---|---|---|---|---|
| **Path 3 (`W8A16-CHANNEL`)** | 7,342 W8A16 (weight-only probe) | **0.1369** | −0.0364 (`t = −57.9`) | **84.05%** | **29.04** | 185.9 / 176.5 / 174.1 / 164.6 |
| **Path 4 ([`W8A8-EXPERTS`](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS), primary ship)** | **7,296 experts W8A8 + 46 attn/dense W8A16** | **0.1527** | **−0.0236 (`t = −36.0`)** | **82.91%** | **28.88** | **200.4 / 189.9 / 187.8 / 176.9** |
| Liquid official FP8 | FP8 weight-only on SM86 | 0.1818 | baseline | 81.18% | 29.29 | 184.5 / 175.0 / 173.4 / 163.5 |
| Path 1 (`W8A8`) | 7,342 W8A8 (all Linears) | 0.2161 | +0.0265 (`t = +36.8`) | 79.72% | 29.21 | 197.2 / 187.4 / 185.1 / 174.5 |
| Path 2 (`SQ-W8A8`, pre-fix) | SmoothQuant + W8A8 (unscaled router bug) | 0.7042 | — | 65.10% | 36.72 | — |

## Expected output size

| | |
|---|---|
| BF16 source | 47.94 GB |
| W8A8 INT8 (this pipeline) | **~24.6 GB** |
| Liquid's official FP8 checkpoint | 24.75 GB |

Nearly identical size. Against the FP8 download: W8A8 INT8
uses real INT8 tensor cores on SM86 through vLLM's Triton Int8 MoE kernel,
whereas FP8 on Ampere serves weight-only (BF16 activations, no INT8 compute
path) — a memory win, not a compute win. Download the built checkpoint at
[Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS),
or rebuild it from BF16 source with this pipeline.

## Memory rules (all paths)

- Accelerate-split load: `device_map="auto"` with per-GPU caps
  (`--gpu-mem`, default 14 GiB — keep well below 24) + 40 GiB CPU + disk
  overflow (`<dst>.offload`). Stop any serving job first so the ~48 GB VRAM
  is actually free.
- `_force_moe_linearize_cpu()` in both quantize scripts pins
  llmcompressor's MoE-linearization workspace to CPU. CPU pinning changes
  placement only, never values.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before torch
  imports in every CUDA entry point.
- `capture-lfm2.py` takes the same memory caps (an uncapped capture would OOM
  on a 48GB source).

## Layout

```
scripts/lfm2_common.py           geometry + IGNORE_RTN/IGNORE_SQ + E12 + census
scripts/quantize_lfm2_rtn.py     Path 1 (W8A8), Path 3 (W8A16-CHANNEL) & Path 4 (W8A8-EXPERTS)
scripts/capture-lfm2.py          Path 2 step 1: activation capture
scripts/lfm2_alpha.py            Path 2 step 2: (base, alpha) grid (defective)
scripts/apply-smoothing-lfm2.py  Path 2 step 3: exact folds (plan_ops/apply_ops)
scripts/quantize_lfm2_sq.py      Path 2 step 4: RTN-E12 applier + census
scripts/smoke.py                 structural check on a quantized dir
tests/test_lfm2_common.py        geometry + recipe unit tests (no model load)
tests/test_fold_plan.py          fold wiring + parity tests (no model load)
tests/test_rtn_scheme.py         scheme selection, overwrite guard, preflight
calib/calib_mixed.txt            calibration corpus (Path 2)
docs/recipe.md                   recipe contract (read before changing it)
docs/accuracy-suggestions.md     post-mortems, probe & Path 4 results
AGENTS.md                        agent context / do-not list
eval/serve_smoke.sh              vLLM serve smoke
eval/composes/                   vLLM serve composes (fp8/sq/w8a16/w8a8/w8a8-experts)
```

Model paths auto-resolve per host (`$LFM2_MODELS`, else `~/models`) — pass
`--src/--dst/--model/--out` only to override.

## Try

```bash
# no local .venv here — use the shared one (has torch/pytest/llmcompressor)
../.venv/bin/python -m pytest tests -q
../.venv/bin/python scripts/quantize_lfm2_rtn.py --dry-run
../.venv/bin/python scripts/quantize_lfm2_rtn.py --dry-run --scheme W8A16-CHANNEL
../.venv/bin/python scripts/quantize_lfm2_rtn.py --dry-run --scheme W8A8-EXPERTS
../.venv/bin/python scripts/capture-lfm2.py --dry-run
../.venv/bin/python scripts/lfm2_alpha.py --dry-run
../.venv/bin/python scripts/apply-smoothing-lfm2.py --dry-run
../.venv/bin/python scripts/quantize_lfm2_sq.py --dry-run
../.venv/bin/python scripts/smoke.py --model ~/models/LFM2.5-24B-A2B-W8A8
../.venv/bin/python scripts/smoke.py --model ~/models/LFM2.5-24B-A2B-W8A16-CHANNEL
../.venv/bin/python scripts/smoke.py --model ~/models/LFM2.5-24B-A2B-W8A8-EXPERTS
```

## References & Credits

- `docs/recipe.md` is the contract; `AGENTS.md` is the do-not list. The publication
  splash page lives on Hugging Face: [Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS). Read `recipe.md` and
  `AGENTS.md` before touching recipes, ignore lists, or the calibration mix.
- **[`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8)**
  — primary methodology and evaluation reference: source of the teacher-forced
  full-vocab KLD + floor-free paired evaluation harness
  (`shared/benchmark-composes/scripts/kld-fullvocab.py` and
  `kld-paired-floorfree.py`), the component-precision approach for isolating
  where INT8 activation error lives, the mixed-precision `compressed-tensors`
  `config_groups` pattern (`int-quantized` W8A8 + `pack-quantized` W8A16), the
  E12 (`down_proj` at W8A16) hypothesis, and the offline SmoothQuant proxy
  design.
- **[`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)**
  — 47.94 GB BF16 base checkpoint and model architecture (`Lfm2MoeForCausalLM`).
- **[`LiquidAI/LFM2.5-24B-A2B-FP8`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-FP8)**
  — official vendor FP8 checkpoint and source of the published keep-out list
  (`lm_head`, `embed_tokens`, `re:.*\.gate$`, `re:.*\.(?:conv|short_conv)\..*`)
  mirrored across Paths 1–4.
- `../shared/` holds the shared venv recipe and eval harness (KLD/extract scripts + corpus).
