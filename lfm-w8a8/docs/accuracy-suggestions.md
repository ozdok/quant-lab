# LFM2.5-24B-A2B INT8 — build log, measurements and open questions

`docs/recipe.md` is the contract; this file is the dev diary behind it — what
was tried, what it measured, and what is still open. It stays in the order the
work happened, so the headline result is not at the top: the shipped build is
**Path 4** (`W8A8-EXPERTS`, § "Path 4 — W8A8-EXPERTS selective split"), and it
beats the vendor FP8 checkpoint on every accuracy metric measured while
running 8.2-8.6% faster decode.

Throughout: **the build node** is the GPU host that quantizes, serves, and
captures KLD; **the client host** drives the speed/extract runs and never
serves a model.

## Notation and conventions

Terms used from here on without further explanation. Nothing else in the repo
defines the fold names — `AGENTS.md` describes A and B in passing, but this is
the only place all four are stated.

**Folds.** Smoothing rewrites that move a per-channel scale off an activation
and into the weights on either side of it, so the quantizer sees a flatter
tensor. Planned in `scripts/apply-smoothing-lfm2.py::plan_ops()`; each fold's
algebra is pinned in `tests/test_fold_plan.py`. They are independent and
selected with `--folds`.

| fold | scale leaves | scale enters | scope |
|---|---|---|---|
| **A** | `ffn_norm` output | every consumer of it — the router (`feed_forward.gate`) and each expert's `w1`/`w3` inputs | layers 2-39 |
| **B** | `operator_norm` output | attention `q_proj`, `k_proj`, `v_proj` inputs | the 10 attention layers |
| **C** | each expert's up projection | `w3` rows out, `w2` (down) columns in | layers 2-39, per expert |
| **E** | each `v_proj` output | `out_proj` input columns, GQA-expanded | the 10 attention layers |

**Paths.** Successive build attempts, in the order they were tried.

| path | scheme | outcome |
|---|---|---|
| **Path 1** | `W8A8` — RTN (round-to-nearest), all 7,342 Linears | shipped first; the baseline later paths are measured against |
| **Path 2** | `SQ-W8A8` — SmoothQuant on Path 1's weights | regressed against Path 1 (KLD 0.7042 vs 0.2161), retired; see the post-mortem |
| **Path 3** | `W8A16-CHANNEL` — per-channel INT8 weights, no activation quantizer | the probe that priced activation quantization |
| **Path 4** | `W8A8-EXPERTS` — the 7,296 experts W8A8, 46 attention/dense Linears W8A16 | **the shipped build** |

**Alpha** is `scripts/lfm2_alpha.py` — the offline proxy that computes the
per-channel scale vectors the folds move, from calibration rows, before the
quantizer runs. It neither quantizes nor produces a checkpoint.

**Measurement levels.** *Matmul level* is one Linear in isolation, fed
calibration rows. *Block level* is a whole model block — a MoE block or an
attention block — end to end. "Block" elsewhere (FP8 weight blocks) means
something unrelated.

**Error.** Unless stated otherwise, error is the relative squared error of a
tensor against its unquantized self — `‖ŷ − y‖² / ‖y‖²` — reported as a
percentage. It is dimensionless, so it is only meaningful next to a named
baseline and a named level (matmul or block).

**Conventions.** KLD is teacher-forced, full-vocab, against the offline BF16
capture, over 240 × 512-token WikiText-2 windows (511 scored positions each,
top-K 512). Logs are natural; everything is in nats. `q floor hits` are
positions where a reference id is absent from an arm's stored top-K, so its
tail mass is spread uniformly across the rest of the vocabulary — the floor
assumption that `kld-paired-floorfree.py` removes.

## Design review (resolved during development)

Six items raised in a design review of this pipeline, each with its status as
it now stands. Items marked *Outstanding* are still true.

1. **C (moe-down) fold must divide the up side only — scaling the gate side
   is inexact through SiLU.** *Impossible by construction.* LFM2 ships gate
   (`w1`), up (`w3`) and down (`w2`) as separate tensors, so there is nothing
   to pair: the fold scales `w3` rows only, which is exact through
   `silu(gate) * up`.
   `tests/test_lfm2_common.py::test_up_rowdiv_only_is_exact_for_swiglu`
   pins both the correct algebra and the inexactness of the fused-halves
   approach, so a regression would fail loudly. Still outstanding: a parity
   test on real weights (folded vs unfolded forward, ~1e-3 relative error in
   BF16, the expected tolerance).
2. **Sampling bias in capture.** *Addressed.* `capture-lfm2.py` picks its
   calibration windows by striding through the corpus instead of taking the
   first `n`. The alpha proxy still takes the first 4096 rows of each hook
   (`PROXY_ROWS`) — stride that the same way if the statistics look
   domain-skewed.
3. **Audit what "mixed" means in the corpus.** *Outstanding.*
   `calib/calib_mixed.txt` is Wikipedia-heavy; LFM2.5 is a reasoning/agentic
   model, so top up instruction/chat/code content before capture.
4. **Census only checks experts.** *Addressed.* The census — the post-build
   count of which tensors landed in which scheme — now counts experts,
   attention, and dense separately and asserts all of `CENSUS_W8A8` /
   `CENSUS_W8A16`. A mis-quant of the attention half can no
   longer print "census OK".
5. **Concrete-class fallback.** *Addressed.* Both quantize scripts import
   `Lfm2MoeForCausalLM` directly and let a failure raise.
6. **Capture had no memory caps.** *Addressed.* `capture-lfm2.py` takes
   `--gpu-mem` / `--offload` and uses the same accelerate split as the
   quantizers.

## Model-specific findings (folds and quantizer behaviour)

1. **The B fold and q/k layernorms.** Resolved 2026-10-17: fold q, k AND v
   (fp32 parity through HF's `Lfm2MoeAttention`: 4e-7; v-only: 21% off). See
   `docs/recipe.md` open question 1.
2. **Zero-centered norms.** `Lfm2MoeRMSNorm` is plain (`w * x`). A
   zero-centered fold (`(1+w)/s - 1`) would be silently wrong here.
   Implemented as
   `norm_plain` (`w / s`); keep it that way.
3. **The 30 conv layers are the biggest untouched block** (0.504B params,
   2.1%). Liquid keeps them at source precision and so does this pipeline. If
   VRAM ever gets tight, smoothing `conv.in_proj` / `conv.out_proj` (they are
   ordinary Linears) is the next lever — but verify coherency, because the
   conv path is what makes this architecture unusual.
4. **`expert_bias` is F32 and must stay there.** It is added to the sigmoid
   routing scores before top-4; quantizing it would perturb expert selection.
   It is a buffer, not a Linear, so `targets=["Linear"]` cannot reach it —
   but do not "helpfully" fold it into the router later.
5. **Verify against Liquid's official FP8 build.** `LiquidAI/LFM2.5-24B-A2B-FP8`
   is 24.75 GB with a published compressed-tensors recipe. Two useful
   cross-checks: (a) it can serve as a quality reference *without* being the
   artifact you ship, and (b) its ignore list
   (`lm_head`, `embed_tokens`, `re:.*\.gate$`, `re:.*\.(?:conv|short_conv)\..*`)
   is an authored second opinion on Path 1's keep-outs.

## Eval policy

Quantize **from** BF16 (`LiquidAI/LFM2.5-24B-A2B`). Candidate builds are judged
on two legs plus a human eval:
1. **Statistical leg (the build node):** teacher-forced full-vocab KLD and floor-free
   paired comparison (`kld-fullvocab.py`, `kld-paired-floorfree.py`, adapted
   from [`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8))
   over 240 × 512-token WikiText-2 test windows (122,640 scored positions — 511
   per window, since the last token has no next-token target) against the
   offline BF16 reference (`kldfv-lfm-bf16.npz`) and official FP8
   (`LiquidAI/LFM2.5-24B-A2B-FP8`).
2. **Functional + speed leg (the client host -> the build node):** sequential
   4-URL arXiv HTML extraction and structured summarization (`extract_test.py`,
   `temperature=0`, `--enable-thinking`, 5.5K to 50.6K prompt tokens — one
   paper per request, fetched then summarized, run sequentially) plus a blind
   quality verdict. Note `--enable-thinking` is inert on this model: LFM2.5 always
   thinks (its chat template has no `enable_thinking` variable and
   unconditionally emits `<|im_start|>assistant\n`), and the trace is split
   into `message.reasoning` by the server's `--reasoning-parser qwen3`, which
   defaults `chat_template_kwargs.enable_thinking` to `True`
   (`vllm/parser/qwen3.py`). Verified 2026-10-18 by A/B: no-kwarg and `true`
   returned identical `reasoning_tokens` (261), while `false` returned 0 and
   moved the trace into `content`. All four arms passed the same flag, so the
   comparison is unaffected.

`smoke.py` proves layout without loading tensor data — it asserts that all 7296
experts are quantized (accepting both `int-quantized` `.weight` `I8` and
`pack-quantized` `.weight_packed` `I32`), that the router, conv block,
embeddings and norms stay unquantized, that expert names are restored to
`w1/w3/w2`, and that no foreign-family tensors (`shared_expert` / `linear_attn` /
`mtp` / `visual`) leaked in.

## Path 2 build log (2026-10-16/17) — how not to do it

Path 2 was staged but had never executed. First full run surfaced seven
latent bugs, all fixed same night:

1. `capture-lfm2.py --dry-run` fell through into model load (missing
   return).
2. Capture used `mm`/`offload` without building them (NameError on any
   real run). Built from `--gpu-mem`/`--offload` per the SQ script idiom.
3. `ModelClass` never imported (leftover `MODEL_CLASS_IMPORT` string).
   Direct `Lfm2MoeForCausalLM` import with CausalLM fallback.
4. Window builder missing entirely (`wins` undefined). Added strided
   selection per recipe.
5. Output dir never created (`calib/acts`). Added makedirs; cost 16
   capture windows of compute.
6. Per-expert `w2` hooks do not exist after HF loads the checkpoint — the
   expert weights arrive as fused 3-D parameters, so there is no per-expert
   Linear to attach a hook to. Broadcasting one expert's statistics to the
   others is dimensionally invalid (the routed input lives in 2048-space, the
   down projection's scale vector in 1536-space). Final: the alpha proxy
   recomputes each expert's down-input in exact BF16 from the shared
   calibration rows plus the stored `w1`/`w3`, takes per-expert amaxes, and
   pools them into one proxy value for the grid cell.
7. Smoothing looked up `self_attn.o_proj`; the real name is `out_proj`.
   (The alpha proxy saved it under the right name; the smoothing script used
   the Qwen-family `o_proj` spelling.)

Two structural findings, not bugs:

8. vLLM (0.29.0) refuses mixed-scheme MoE: all expert projections must
   share one quantization scheme. E12 (Freaksterz's hypothesis: the experts'
   down projection at W8A16, gate/up at W8A8) builds fine and passes census
   (4908/2434) but cannot serve. Freaksterz never hit this
   because Qwen3.8 is dense. Pivot:
   uniform RTN on smoothed weights (`LFM2.5-24B-A2B-SQ-W8A8`), which
   serves normally.
9. Smoothed-uniform regressed vs Path 1 RTN: KLD 0.70 vs 0.22, top-1
   65% vs 80% vs BF16, PPL 29.32 -> 36.72 (Path 1: 29.21). Extracts
   stayed coherent, which is why both legs exist. Diagnosed 2026-10-17:
   fold A left the router unscaled (see the post-mortem below).

## Path 2 post-mortem (2026-10-17) — why KLD grew

Verdict: the SQ-W8A8 regression was a wiring bug, not a SmoothQuant failure.
Everything below was measured read-only on the build node (calibration rows in
`calib/acts`, the BF16 source, and the four `kldfv-lfm-*.npz` captures). The
checks were throwaway scripts; `tests/test_fold_plan.py` now pins their
structural half.

### 1. Root cause: fold A left the router unscaled

`ffn_norm`'s output feeds the router (`feed_forward.gate`) and all 64
experts. Fold A divided `ffn_norm` by `s_a` and multiplied each expert's
`w1`/`w3` input columns by `s_a`, but the planner never touched the router, so
in the smoothed checkpoint the router saw `x / s_a`.

- Routing, 12,288 calibration rows x 38 MoE layers: the top-4 expert set
  changed for 62% of tokens (mean slot overlap 80.5% — a slot is one of the
  four selected expert positions — worst layer ~70%).
  Scaling the gate columns by `s_a` restores 99.87% slot agreement; the rest
  is BF16 rounding of the folded gate.
- `s_a` was never normalised (geometric mean 1.1 in layer 2, 5.6 in layer 38),
  so deep-layer router logits shrank to ~0.2x: sigmoid scores flattened and
  `expert_bias` dominated selection.
- Equivalence before any quantization (fp32, layer 3, real weights): MoE block
  output 21.3% off as built, 3.3e-7 off with the router folded.
- With INT8 emulated at block level (per-token INT8 activations and
  per-channel INT8 weights simulated in PyTorch on the real calibration rows —
  no checkpoint built; layers 3 / 12 / 24 / 37): plain RTN 3.7 / 3.6 / 3.9 /
  3.5%; as built 21 / 35 / 34 / 53%; router fixed but old scales 3.9 / 4.2 /
  5.1 / 3.6%.
- The KLD captures agree: the damage is broad (SQ's best-10% window, 0.52, is
  worse than RTN's 90th-percentile window, 0.33), it costs +0.225 nats NLL on
  real text against ~0 for RTN and FP8, and position-0 top-1 agreement is 4.6%
  vs 48%.

### 2. Second mistake, also fixed: fold C nested under the expert loop

The `--folds` patch (2026-10-18 00:39 UTC) de-indented the C block out of
`for e in range(N_EXPERTS)`, so C hit expert 63 only (38 tensors instead of
2432) and `--folds C` planned zero tensors (a silent copy of the source). The
0.70 build predates the patch, so it did not cause the regression; it would
have invalidated the queued sweep. The planner now lives in `plan_ops()` with
wiring tests, and a tautological `s_a` shape assert became a real one.

### 3. Verified exact (fp32)

Parity is measured **against** HF's own `Lfm2MoeAttention` on layer 2's real
weights — that module is the yardstick here, not a suspect:

- B (q, k AND v columns): 3.9e-7. E (GQA v rows / out columns): 4.6e-7.
  B+E: 5.3e-7.
- A+C with the router folded, layer 3: 3.3e-7.
- BF16 re-rounding of the folded weights: 0.2% block error with routing forced
  to the reference, 0.5-1% with free routing (near-tie router flips).

Two **negative controls** — deliberately broken variants of our own test block,
run to prove the check can actually fail. Both are our errors, never HF bugs:

- B without q/k ("v-only", the older advice): 21% off.
- GQA expansion written as `repeat` instead of `repeat_interleave`: 40% off.
  Only the `out_proj` half of E is GQA-sensitive — each kv head's scale vector
  must repeat `G` times *in place* to line up with `out_proj`'s columns. This
  control is what pins our head ordering to HF's `repeat_kv`.

### 4. Still open: the scale search (`lfm2_alpha.py`)

On held-out rows the incumbent scales were worse than plain RTN (identity) for
fold A in 26 of 38 layers (mean -35%, worst -218% at layer 17), fold C in 16 of
16 sampled experts, and fold E (mean -22%); only B helped (about +39%). A
corrected protocol (identity candidate; 99.9th-percentile stats of typical
rows; first-token rows excluded; selection on windows 0-63, scoring on windows
64-95, of the 96 calibration windows) gave, at matmul level: A +39% (0 of the
11 sampled layers worse), B +66%, E +3%. Percentages in this section are
relative reductions in the alpha proxy's error (`X/s` and `W·s` quantized
per-token and per-channel respectively,
`err = ‖q(X/s)·q(W·s)ᵀ − X·Wᵀ‖²_F / ‖X·Wᵀ‖²_F`) measured against the identity
(RTN) scales. Fold B reproduces on the held-out windows at +40.0% against the
+39% quoted here; the same scales read −40.0% against the old grid's
`(absmax, 0.8)` reference cell, which is why the baseline has to be named. C
added nothing at block level (3.71 / 3.56% vs RTN 3.69 / 3.52%). At block level
with routing forced to the reference, corrected A cut the error 3.69 -> 3.58%
(layer 3) and 3.52 -> 2.73% (layer 24); with free routing the BF16 re-rounding
noise above erased that (3.8% vs 3.5% at layer 24).

Ruled out as causes: first-token massive activations (0.5-1.1% of output
power, about their 0.78% share of rows) and overfitting to the four sampled
experts (+43-52% on unseen experts too).

### 5. The yardstick: what "match FP8" means

| | KLD vs BF16 | top-1 | PPL (BF16 29.32) | mean dNLL |
|---|---|---|---|---|
| Liquid FP8 | 0.182 | 81.2% | 29.29 | -0.001 |
| Path 1 RTN | 0.216 | 79.7% | 29.21 | -0.004 |
| Path 3 W8A16-CHANNEL probe | 0.137 | 84.1% | 29.04 | -0.010 |
| SQ-W8A8 (broken) | 0.704 | 65.1% | 36.72 | +0.225 |

This table is a rounded restatement; the same arms at full precision, with the
paired bootstrap, are in "Path 3 — W8A16-CHANNEL probe" -> Result.

FP8 vs W8A8 is 0.241 apart, larger than either arm's distance to BF16, so the
metric has a high floor on this model; positions under 16 carry ~4-7x the KLD
of positions past 256. Block-level error budget for plain RTN (layers 3 / 24):
weights alone ~1.0%, input-activation quantization 1.3 / 2.7%, down-input
activation quantization 3.3 / 2.0%, all together 3.69 / 3.52% — the parts are
measured separately, so they do not sum to the combined figure. FP8 on Ampere
runs as W8A16 (no activation quantization); the W8A16-CHANNEL probe below
measured how much of the gap that is: all of it. E12 targeted the same lever
but is blocked by vLLM's uniform-scheme wall.

## Path 3 — W8A16-CHANNEL probe (2026-10-17/18) — what the activation quantizer costs

Question: how much of Path 1's gap to Liquid's FP8 (KLD 0.216 vs 0.182) is
activation quantization versus weight rounding? FP8 runs weight-only on
Ampere, so the clean test is Path 1's exact weights with the activation
quantizer removed, served in the same vLLM and captured on the same 240
windows.

### Method, and two traps

- `quantize_lfm2_rtn.py --scheme W8A16-CHANNEL` (about 9.5 min). Verified
  afterwards: the same 7342 quantized modules as Path 1, 268 unquantized
  tensors identical in name/dtype/shape, and every sampled weight (8: experts,
  attention, dense) bit-identical to Path 1's after unpacking int32 -> int8,
  with identical BF16 scales.
- Trap 1: the stock compressed-tensors 0.18 `W8A16` preset is GROUP-128, finer
  than Path 1's per-channel weights. It would have flattered the weight error
  and corrupted the decomposition, so the probe reuses Path 1's weight args
  under its own name. `assert_saved_scheme` and a recipe test pin this.
- Trap 2: vLLM 0.29.0 cannot initialise a per-channel W8A16 MoE checkpoint as
  saved. The MoE WNA16 backend oracle compares the raw `group_size`
  (`marlin_utils.check_moe_marlin_supports_config`: `group_size <= 0`) before
  normalising it, so `group_size: null` raises TypeError during engine init.
  `group_size: -1`, compressed-tensors' channel-wise convention, fixes it. The
  script now writes it (`set_channel_group_size`); the existing probe directory
  got the same single-field edit by hand (reverting it reproduces the original
  config byte-for-byte). vLLM then routed the weight-only MoE experts through
  `CompressedTensorsWNA16MoEMethod` -> the WNA16 MoE oracle
  (`fused_moe/oracle/int_wna16.py`) -> **MARLIN** (`MarlinExperts`), and the
  weight-only dense/attention Linears through `CompressedTensorsWNA16` ->
  **AllSpark** (`AllSparkLinearKernel`).
- Serve flags are identical to the FP8 arm: `docker-compose.lfm-w8a16.yml` is
  the FP8 compose with only names, cache dirs and the model path changed.

### Result

| | KLD vs BF16 | top-1 | PPL (BF16 29.32) | mean dNLL |
|---|---|---|---|---|
| Liquid FP8 | 0.1818 | 81.18% | 29.29 | -0.001 |
| Path 1 W8A8 | 0.2161 | 79.72% | 29.21 | -0.004 |
| W8A16-CHANNEL | 0.1369 | 84.05% | 29.04 | -0.010 |

Every quantized arm scores slightly *better* than the BF16 reference on PPL and
mean dNLL. Read that as the runtime floor, not as a quantization gain: the
reference is captured offline while the arms are served by vLLM (see "Reading
it" item 4), and that gap is larger than the differences here.

Paired bootstrap over the 240 windows (positions are correlated inside a
window, so windows are the independent unit). This is the **full-support
(floored)** statistic — the same one as the table above, so each delta equals
the difference of the corresponding means there:

| paired difference | KLD | 95% CI | top-1 |
|---|---|---|---|
| W8A8 - FP8 (the gap) | +0.034 | [+0.022, +0.047] | -1.46 pt |
| W8A8 - W8A16 (activation quantization) | +0.079 | [+0.069, +0.089] | -4.33 pt |
| W8A16 - FP8 (INT8 vs FP8 weights) | -0.045 | [-0.055, -0.034] | +2.87 pt |

Both READMEs and the model card quote the **floor-free** equivalents of the
first and third rows — Path 1 vs FP8 `+0.0265` (`t = +36.8`) and Path 3 vs FP8
`-0.0364` (`t = -57.9`). They are smaller in magnitude than the rows above for
the reason given below. Both statistics are correct, and they are not
interchangeable.

`kld-paired-floorfree.py` (the floor assumption removed) agrees: W8A16 beats
FP8 (t = +57.9, better at 66.9% of positions) and FP8 beats W8A8 (t = +36.8).
That check was required: the `q floor hits` rates (15.6 / 18.3 / 19.8%) track
the KLD ranking, the pattern that script warns about, and the ranking held.
The probe's advantage over FP8 is uniform across position buckets (its KLD is
0.65-0.80x FP8's in every bucket), and so is the activation cost (W8A8's KLD is
1.5-1.75x W8A16's), 91% of it past position 16.

### Reading it

1. The whole gap to FP8 is activation quantization. INT8 per-channel weights
   are MORE accurate than FP8's (consistent with e4m3 keeping 3 mantissa
   bits), so the W8A16 arm overshoots FP8 by 0.045 KLD / 2.9 pt while
   activations cost 0.079 KLD / 4.3 pt.
2. To match FP8, W8A8 has to recover about 43% of its activation cost on KLD
   (0.034 of 0.079) and about 34% on top-1 (1.46 of 4.33 pt). KLD is only
   approximately additive at these magnitudes: treat both as rough targets.
3. Smoothing's measured block-level ceiling (A -22% at layer 24 with routing
   held fixed and nothing with free routing; B small; E +3%; C flat) makes a
   34-43% cut from smoothing alone unlikely. Not measured end to end.
4. Floor: the three vLLM-vs-vLLM KLDs (FP8-W8A8 0.241, FP8-W8A16 0.192,
   W8A8-W8A16 0.204) fit an additive model `KLD(i,j) = shared + noise_i +
   noise_j`. That is three equations in four unknowns, so the triangle pins
   the arm-to-arm differences only and the shared term stays free. Its
   shared = 0 solution is the arm noise quoted here — 0.13 (W8A8), 0.11 (FP8),
   0.08 (W8A16) — and any nonzero shared term subtracts from all three alike
   (0.1265 - s, 0.1145 - s, 0.0775 - s); requiring non-negative arm terms
   bounds the shared component at 0-0.08 nats. Separating the two needs a
   direct same-arm HF-vs-vLLM capture, which nothing in this document measures.
   Absolute KLD on this model is not comparable with published figures; only
   paired differences are.
5. By-product: Path 3 (`W8A16-CHANNEL`) is a servable checkpoint that beats FP8
   on every metric measured — both accuracy (0.137 KLD / 84.1% top-1 / 29.04
   PPL) and speed (+0.4-0.9% decode TPS and 2.7-3.6% faster TTFT than FP8 across
   all four arXiv extracts: 185.9 / 176.5 / 174.1 / 164.6 tok/s vs FP8's 184.5 /
   175.0 / 173.4 / 163.5). It has no INT8
   compute path, so it serves as our INT8 weight-only baseline on Ampere.

### Where this leaves the plan

1. B-only is now a pipeline validation (does the fixed chain serve and not
   regress), not a gap-closer.
2. The corrected scale search plus fold A is the only smoothing arm with
   measured headroom (A +39% at matmul level, -22% block error at layer 24 with
   routing held fixed), and it is router-noise-limited.
3. Selective component splits starting from Path 3 (staged as **Path 4**,
   `--scheme W8A8-EXPERTS`: moving MoE experts to W8A8 while keeping attention
   and dense layers 0-1 at W8A16; see section below) are the clean next lever:
   every component stays uniform inside its own fused unit, and Path 3's
   accuracy cushion gives room to buy W8A8 speed.
4. Speed measured 2026-10-18 (the client host -> the build node), all four arms across the 4
   arXiv URLs — decode tok/s: Path 4 200.4 / 189.9 / 187.8 / 176.9,
   Path 1 197.2 / 187.4 / 185.1 / 174.5, Path 3 185.9 / 176.5 / 174.1 / 164.6,
   FP8 184.5 / 175.0 / 173.4 / 163.5. Captures:
   `testing/lfm-{fp8,w8a8,w8a16,w8a8-experts}/`.
5. Floor capture (HF eager vs sdpa) is lower priority now: Path 3 and the
   triangle above already bound it.

## Path 4 — W8A8-EXPERTS selective split (built & measured 2026-10-18)

### Reasoning

Path 3 (`W8A16-CHANNEL`) established two facts:
1. Per-channel INT8 weights beat Liquid's FP8 weights by **0.045 KLD** and
   **+2.87 pt top-1** (0.137 / 84.05% vs 0.182 / 81.18%), while running ~1%
   faster on decode and ~3-4% faster on TTFT.
2. The entire gap between Path 1 (`W8A8`, 0.216 / 79.72%) and FP8 is activation
   quantization (+0.079 KLD / -4.33 pt top-1 across all 7342 quantized Linears).

Path 1 quantizes activations in three distinct component classes:
- **MoE routed experts** (`layers.2..39.feed_forward.experts.{0..63}.w{1,2,3}`):
  **7296** Linears, **22.95B params (95.7%)**. Active 4-of-64 per token; drives
  nearly all of Path 1's +7% decode throughput advantage over weight-only.
- **Full-attention projections** (`layers.{2,6,..,38}.self_attn.{q,k,v,out}_proj`):
  **40** Linears, **0.105B params (0.4%)**. Runs on every token across all 10
  attention layers.
- **Early dense MLPs** (`layers.{0,1}.feed_forward.w{1,2,3}`):
  **6** Linears, **0.145B params (0.6%)**, intermediate size 11776. Layers 0 and
  1 execute unconditionally on every token and shape the initial residual stream
  that feeds all 38 downstream MoE routers and experts.

Path 4 (`--scheme W8A8-EXPERTS`) keeps the 40 attention Linears and 6 dense
layers 0-1 Linears at Path 3's `W8A16` (INT8 per-channel weights, BF16
activations) while moving all 7296 MoE expert Linears to `W8A8` (INT8
per-channel weights, dynamic per-token INT8 activations). This tests two
hypotheses in a single build:
- **Accuracy:** how much of the 0.079 KLD activation cost came from the early
  dense MLPs (layers 0-1) + attention vs the MoE experts themselves.
- **Speed:** whether putting 95.7% of parameters (all routed experts) on the
  INT8 tensor-core fast-path captures Path 1's decode speedup while preserving
  any accuracy share from BF16 activations in layers 0-1 and attention.

### Why Path 4 serves in vLLM 0.29.0 (where E12 failed)

E12 failed at vLLM startup because it put `w1/w3` in W8A8 and `w2` in W8A16
*within the same MoE expert*. `CompressedTensorsMoEMethod.get_moe_method` queries
`.0.gate_proj`, `.0.up_proj`, and `.0.down_proj` on each `RoutedExperts` layer
and raises `ValueError("All MoE projections need to have same quantization
scheme but found multiple")` if they differ.

Path 4 respects every fusion boundary in vLLM's `Lfm2MoeForCausalLM`:
- **`RoutedExperts` (`layers.2..39.feed_forward.experts`):** all three
  projections (`w1`, `w3`, `w2` / `gate_proj`, `up_proj`, `down_proj`) match
  `group_0` (`PATH4_W8A8`, `int-quantized` with dynamic per-token input
  activations) -> method `CompressedTensorsW8A8Int8MoEMethod`, executed by the
  **Triton Int8 MoE kernel** (`TritonExperts`; vLLM logs
  `Using TRITON Int8 MoE backend out of potential backends: ['TRITON',
  'HUMMING', 'CPU']`).
- **Attention `qkv_proj` (`q_proj`, `k_proj`, `v_proj` stacked) & `out_proj`:**
  all shards match `group_1` (`PATH4_W8A16`, `pack-quantized` weight-only) ->
  method `CompressedTensorsWNA16`, executed by the **AllSpark** linear kernel.
- **Dense MLP `w13` (`w1`, `w3` stacked) & `w2` in layers 0-1:** all shards
  match `group_1` (`PATH4_W8A16`, `pack-quantized` weight-only) -> method
  `CompressedTensorsWNA16`, executed by the **AllSpark** linear kernel.
- **Router (`feed_forward.gate`) & short-conv (`conv.*` / `short_conv.*`):** in
  `IGNORE_RTN` -> unquantized BF16.

Verified directly against vLLM 0.29.0's `CompressedTensorsConfig` resolver on
the build node before staging.

### Execution runbook

1. **Precondition:** stop any running container on the build node (`docker ps` empty on
   `:8051`; ~48 GB VRAM free).
2. **Dry-run & tests:**
   ```bash
   cd <repo>/lfm-w8a8
   ../.venv/bin/python -m pytest tests -q
   ../.venv/bin/python scripts/quantize_lfm2_rtn.py --scheme W8A8-EXPERTS --dry-run
   ```
3. **Build (~10 min on the build node):**
   ```bash
   ../.venv/bin/python -u scripts/quantize_lfm2_rtn.py --scheme W8A8-EXPERTS \
     > logs/path4-w8a8-experts.log 2>&1
   ```
   Verify `census OK` (7296 experts W8A8, 40 attn W8A16, 6 dense W8A16,
   `mismatch: []`), `expert names restored: 14592 tensors -> w1/w3/w2`
   (two per quantized Linear — weight and scale — so 7296 × 2),
   `channel-wise group_size -> -1`, and `saved scheme verified: W8A8-EXPERTS`.
4. **Serve & warm up:**
   Serve `~/models/LFM2.5-24B-A2B-W8A8-EXPERTS` on `:8051` via a benchmark
   compose identical to `docker-compose.lfm-w8a8.yml` (changing only model path,
   cache dir, and container name), then run `warmup.py` from the client host.
5. **Measure:**
   - **KLD leg (the build node):** `kld-fullvocab.py capture-vllm` against
     `eval/results/kldfv-lfm-bf16.npz` (`--vocab 128000`), then `compare` and
     `kld-paired-floorfree.py` against BF16, FP8, Path 1 (`w8a8`), and Path 3
     (`w8a16`).
   - **Speed + extract leg (the client host):** sequential 4-URL arXiv run at temp 0,
     `--enable-thinking` into `testing/lfm-w8a8-experts/`.

### Results (2026-10-18)

#### 1. Statistical leg (KLD / Top-1 / PPL across all 240 windows, 122,640 positions)

| Arm | KLD vs BF16 | Median KLD | Top-1 vs BF16 | PPL (BF16 29.32) | q floor hits |
|---|---|---|---|---|---|
| Liquid FP8 | 0.1818 | 0.0954 | 81.18% | 29.29 | 18.27% |
| Path 1 `W8A8` (all Linears W8A8) | 0.2161 | 0.1180 | 79.72% | 29.21 | 19.83% |
| Path 3 `W8A16-CHANNEL` (all Linears W8A16) | 0.1369 | 0.0652 | 84.05% | 29.04 | 15.63% |
| **Path 4 `W8A8-EXPERTS` (experts W8A8 + attn/dense W8A16)** | **0.1527** (±0.00084) | **0.0749** | **82.91%** | **28.88** | **16.58%** |

Floor-free paired comparisons (`kld-paired-floorfree.py`, K=512). **This is
a different statistic from the table above, not a recomputation of it.** The
script sums only over reference ids present in *both* arms' supports, so the
retained probability mass does not sum to 1 and the result is a comparison
statistic rather than a KL — in its own words, "the absolute level is
meaningless". These deltas are by construction smaller in magnitude than the
differences of the table's KLD column, and subtracting two cells of that table
will not reproduce them. Compare arms within this block only:
- **Path 4 vs Liquid FP8:** paired delta **-0.0236 ± 0.00066** (`t = -36.0`,
  floor-free `-16.7%`, Path 4 better at **60.99%** of positions).
- **Path 4 vs Path 1 `W8A8`:** paired delta **-0.0509 ± 0.00071** (`t = -71.9`,
  floor-free `-30.3%`, Path 4 better at **68.46%** of positions).
- **Path 4 vs Path 3 `W8A16-CHANNEL`:** paired delta **+0.0129 ± 0.00046**
  (`t = +28.2`, floor-free `+12.3%`).

**Key takeaway:** Keeping just **46 Linears** (the 6 early dense MLP Linears in
layers 0-1 + the 40 attention Linears, only **1.0%** of model parameters) at
W8A16 while putting all **7,296 MoE expert Linears (95.7% of parameters)** at
W8A8 eliminates **80.1% of the activation quantization KLD penalty** (`0.0634`
of `0.0792` nats) and recovers **73.7% of the top-1 drop** (`+3.19` of `4.33`
pt), decisively beating Liquid's FP8 (`0.1527` vs `0.1818` KLD, `82.91%` vs
`81.18%` top-1, `28.88` vs `29.29` PPL).

#### 2. Speed + extract leg (the client host -> the build node, sequential, temp 0, thinking on)

| arXiv URL (prompt tokens) | FP8 decode / TTFT | Path 1 `W8A8` decode / TTFT | Path 3 `W8A16` decode / TTFT | **Path 4 `W8A8-EXPERTS` decode / TTFT** |
|---|---|---|---|---|
| `2607.15866v1` (5,525) | 184.5 tok/s / 778 ms | 197.2 tok/s / 653 ms | 185.9 tok/s / 757 ms | **200.4 tok/s / 660.8 ms** |
| `2606.16140v1` (20,888) | 175.0 tok/s / 2923 ms | 187.4 tok/s / 2884 ms | 176.5 tok/s / 2820 ms | **189.9 tok/s / 2919.1 ms** |
| `2603.28052v1` (24,105) | 173.4 tok/s / 3281 ms | 185.1 tok/s / 2752 ms | 174.1 tok/s / 3163 ms | **187.8 tok/s / 2804.8 ms** |
| `2610.22111v1` (50,572) | 163.5 tok/s / 7335 ms | 174.5 tok/s / 6234 ms | 164.6 tok/s / 7090 ms | **176.9 tok/s / 6322.4 ms** |

Across all 4 URLs, Path 4 runs **+8.2% to +8.6% faster decode than FP8**
and **+7.5% to +7.9% faster decode than Path 3 `W8A16`** (and ~+1.5% above Path 1
on the same baseline), while beating Path 3 on
50.6k-prompt TTFT by **767 ms (-10.8%)**.

#### 3. Post-save config group-ordering note (fixed & pinned)

`ModelCompressor.from_pretrained_model` numbers `config_groups` in the order it
first encounters matching modules during `model.named_modules()`. Because
`model.layers.0` (dense MLP, `PATH4_W8A16`) precedes `model.layers.2` (first
MoE layer, `PATH4_W8A8`), the saved `config.json` writes `group_0` =
`pack-quantized` (`PATH4_W8A16`) and `group_1` = `int-quantized` (`PATH4_W8A8`).
`assert_saved_scheme` in `scripts/lfm2_common.py` now matches groups by their
target sets rather than assuming a fixed `group_0`/`group_1` order, pinned in
`tests/test_rtn_scheme.py`.

## References & Artifacts

- **Primary methodology, experiment design & KLD harness:**
  [`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8)
  — teacher-forced full-vocab KLD + floor-free paired comparison scripts
  (`kld-fullvocab.py`, `kld-paired-floorfree.py`), component-level activation
  error isolation ("Where the INT8 activation error lives"), mixed-precision
  `compressed-tensors` `config_groups` (`int-quantized` W8A8 + `pack-quantized`
  W8A16), E12 (`down_proj` at W8A16) hypothesis, and offline SmoothQuant proxy
  heritage (`alpha-proxy.py` / `apply-smoothing-v2.py`).
- **Base BF16 model:**
  [`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
  (`~/models/LFM2.5-24B-A2B`, 47.94 GB, 10 shards).
- **Vendor FP8 reference & ignore-list contract:**
  [`LiquidAI/LFM2.5-24B-A2B-FP8`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-FP8)
  (`~/models/LFM2.5-24B-A2B-FP8`, 24.75 GB).
- **Built checkpoints on the build node (`~/models/`):**
  - Path 1 (`W8A8`): `LFM2.5-24B-A2B-W8A8` (24 GB, all 7,342 Linears W8A8).
  - Path 3 (`W8A16-CHANNEL`): `LFM2.5-24B-A2B-W8A16-CHANNEL` (24 GB, all 7,342
    Linears per-channel W8A16).
  - Path 4 (`W8A8-EXPERTS`): `LFM2.5-24B-A2B-W8A8-EXPERTS` (24 GB, 7,296 MoE
    experts W8A8 + 40 attention + 6 dense layers 0-1 W8A16) — published at
    [Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS).
- **KLD captures (`../eval/results/`):**
  `kldfv-lfm-bf16.npz` (offline reference), `kldfv-lfm-fp8.npz`,
  `kldfv-lfm-w8a8.npz`, `kldfv-lfm-sq.npz`, `kldfv-lfm-w8a16.npz`,
  `kldfv-lfm-w8a8-experts.npz` (240 × 512-token WikiText-2 test windows from
  `shared/benchmark-composes/corpus/wiki.test.raw`).
- **Functional & speed captures on the client host (`testing/` — per-arm run
  transcripts and timings):**
  `lfm-fp8/`, `lfm-w8a8/`, `lfm-w8a16/`, `lfm-w8a8-experts/` (sequential temp-0
  thinking-on runs across `arxiv:2607.15866v1`, `arxiv:2606.16140v1`,
  `arxiv:2603.28052v1`, `arxiv:2610.22111v1`).
