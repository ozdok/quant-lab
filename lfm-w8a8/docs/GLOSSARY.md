# Glossary

Vocabulary used across this model directory's docs — the READMEs, `recipe.md`,
the build log in `docs/accuracy-suggestions.md`, and the model card on
Hugging Face. **This file is the single source of truth for these terms.** If a
definition here and one elsewhere disagree, this file is right and the other is
a bug.

**Scope rule:** this file defines terms; it does not report results. Numbers
live in the measurements that produced them, so nothing here can go stale
against a re-run.

Everything below applies to every family in this repo. Only the geometry table
is model-specific.

## Geometry (LFM2.5-24B-A2B)

| | |
|---|---|
| layers | 40 — layers 0-1 dense MLP, layers 2-39 MoE |
| full-attention layers | 10, every 4th from layer 2 (2, 6, …, 38) |
| gated short-conv layers | the other 30 |
| experts | 64 per MoE layer, top-4 per token, no shared expert |
| widths | hidden 2048 · expert intermediate 1536 · dense intermediate 11776 |
| attention | 32 query heads / 8 KV heads / head_dim 64 (`G` = 4 query heads per KV group) |
| vocab | 128,000, `tie_word_embeddings: true` (no `lm_head` on disk) |
| quantized Linears | **7,342** = 7,296 experts (38 × 64 × 3) + 40 attention (10 × 4) + 6 dense (2 × 3) |

## Quantization schemes

| term | meaning |
|---|---|
| **INT8 weights** | Symmetric 8-bit integer weights, **one scale per output channel** (`strategy: channel`, `group_size: -1` in the `compressed-tensors` config — `-1` is that library's channel-wise convention, not a group size). |
| **INT8 activations** | Symmetric 8-bit integer activations, **one scale per token**, computed at run time from that row's own amax (`strategy: token`, `dynamic: true`). |
| **`W8A8`** | 8-bit weights *and* 8-bit activations. The only scheme that reaches INT8×INT8 tensor cores. |
| **`W8A16`** | 8-bit weights, BF16 activations (`input_activations: null`) — weight-only. Cheaper to serve, leaves the INT8 compute path unused. |
| **`W8A16-CHANNEL`** | The `W8A16` scheme spelled out in `--scheme`, making the per-channel weight axis explicit. Identical to `W8A16`. |
| **RTN** | Round-to-nearest. Data-free quantization: no calibration corpus, no error feedback, no GPTQ-style fitting — every weight rounds independently. This is why the checkpoints can claim no calibration-domain bias is baked in. |
| **smoothing / SmoothQuant (SQ)** | The activation-smoothing technique this pipeline adapts: move a per-channel scale off an activation and into the adjacent weights *before* quantizing, so both sides are flatter. The folds are exact (see below); only the scale *choice* is estimated. |
| **GPTQ** | A calibration-fitted, error-compensating weight-quantization method. **Deliberately not used** — it overfits the calibration corpus, and this pipeline's design choice is data-free weight rounding. |
| **FP8** | Liquid AI's official checkpoint format: 8-bit float (`e4m3`) with 128×128 weight blocks and group-128 dynamic activations. Ampere has no FP8 tensor cores, so vLLM serves it as weight-only — no activation quantization. It is the accuracy reference, never an artifact this pipeline ships. |
| **`compressed-tensors`** | The quantization format and library (`llm-compressor`) that writes the saved `config.json`: named `config_groups`, each with `targets` (regex or name), a weight/activation spec, and a global `ignore` list. |
| **E12** | Freaksterz's hypothesis, adopted as an experiment: within one MoE expert, put gate/up at W8A8 and the down projection at W8A16. Builds and passes the census here, but vLLM 0.29.0 refuses to serve it — see *fused units*. |
| **census** | The build scripts' post-quantization self-check: count the quantized modules per component class (experts / attention / dense) and assert each matches its hard-coded expectation. A mis-quantized component cannot print `census OK`. |
| **smoke test** (`scripts/smoke.py`) | A layout check that never loads tensor data: asserts every expert is quantized (accepting both `int-quantized` `.weight` `I8` and `pack-quantized` `.weight_packed` `I32`), that the router, conv block, embeddings and norms stay unquantized, that expert names were restored to `w1/w3/w2`, and that no foreign-family tensors leaked in. |

## Build paths

Successive build attempts, in the order they were tried. "Path" is this
project's label — the numbers are the only thing that distinguishes them.

| path | scheme | outcome |
|---|---|---|
| **Path 1** | `W8A8` — RTN, all 7,342 Linears | shipped first; the baseline every later path is measured against |
| **Path 2** | `SQ-W8A8` — smoothing folds on Path 1's weights | regressed against Path 1 and was retired; its post-mortem is the largest section of the build log |
| **Path 3** | `W8A16-CHANNEL` — per-channel INT8 weights, no activation quantizer | the probe that measured what activation quantization alone costs. "**The probe**" elsewhere means this. |
| **Path 4** | `W8A8-EXPERTS` — the 7,296 experts at W8A8, the 46 attention/dense Linears at W8A16 | **the shipped build** |

## Smoothing folds

A fold is an **exact rewrite**: it moves a per-channel scale from an activation
into the weights on either side of it, leaving the fp32 computation unchanged.
"Exact" describes the fold itself — error is introduced only by the
quantization that follows. Planned by
`scripts/apply-smoothing-lfm2.py::plan_ops()`; the algebra and the wiring are
pinned in `tests/test_fold_plan.py`. Folds are independent, selected with
`--folds`.

| fold | scale leaves | scale enters | scope |
|---|---|---|---|
| **A** | `ffn_norm` output | **every consumer** of that tensor: the router (`feed_forward.gate`) *and* each expert's `w1`/`w3` inputs | layers 2-39 |
| **B** | `operator_norm` output | all three of `q_proj`, `k_proj`, `v_proj` inputs | the 10 attention layers |
| **C** | each expert's up projection | `w3` **rows** out, `w2` (down) **columns** in — up side only, because the gate side passes through SiLU | layers 2-39, per expert |
| **E** | each `v_proj` output | `out_proj` input columns, **GQA-expanded** (each KV head's scale repeated `G` times *in place*, `repeat_interleave` not `repeat`) | the 10 attention layers |

Folds A and B are the two places where "scale every consumer" is the whole
trick: a norm's output fans out to more than one module, and scaling only some
of them silently changes the function.

## Scale search

| term | meaning |
|---|---|
| **alpha** (`scripts/lfm2_alpha.py`) | The offline script that chooses the per-channel scale vector for each fold. It simulates the fold plus INT8 quantization for every candidate scale on the captured rows, so it needs no build/serve cycle per candidate — hence "proxy". |
| **proxy** | Any offline simulation standing in for a real build-and-serve cycle. |
| **the grid** | The candidate scales: a *base* activation statistic × an *alpha* exponent. `alpha` interpolates between the activation distribution and the weight distribution, as in SmoothQuant. |
| **identity candidate** | A candidate scale of 1 — i.e. plain RTN. The original grid lacked it, which is why its chosen scales could lose to RTN; see the revision spec in `recipe.md`. |
| **`PROXY_ROWS`** | The 4,096-row cap the proxy applies per hook (strided, not head-truncated) to bound VRAM. |

## Measurement

| term | meaning |
|---|---|
| **matmul level** | Error measured on one Linear in isolation (or one stacked projection group), fed real captured rows. |
| **block level** | Error measured on a whole model block — a MoE block or an attention block — run end to end. |
| **error** | Unless stated otherwise, the relative squared error of a tensor against its unquantized self: `‖ŷ − y‖² / ‖y‖²`, as a percentage. Dimensionless, so it means nothing without a named baseline and a named level. |
| **capture / acts** | The calibration capture (`calib/acts/`): 96 windows of the calibration corpus, 2,048 tokens each, run through the BF16 model. Per-window channel maxima are kept for all 58 hooks; activation rows are sampled at 128 strided positions per window (token 0 is always one of them) and sharded every 32 windows. |
| **calibration corpus** | `calib/calib_mixed.txt` — the text the capture runs on. Wikipedia-heavy, which is a known limitation for a reasoning/agentic model. |
| **windows** | The scored units of the KLD harness: 240 × 512-token windows of WikiText-2 test text, taken by striding (not head-truncation). |
| **scored positions** | The positions a window actually contributes: 511 of its 512 tokens, because the last token has no next-token target — 240 × 511 = 122,640 across the harness. |
| **KLD** | The divergence measure, computed as `KL(P‖Q) = Σ p_i (log p_i − log q_i)` summed over the **reference's stored top-K support**. The reference is captured offline with an untruncated softmax (hence "full-vocab") and stores the top-K ids, exact log-softmax logprobs, and the exact tail log-mass; the candidate is captured either offline or teacher-forced through the served vLLM using the reference's own token ids, so both arms are compared at identical positions. |
| **mean KLD / median KLD** | Mean and median of the per-**position** KL over all scored positions. Positions inside a window are correlated, so the mean's standard error understates the uncertainty — the paired bootstrap treats windows as the independent unit instead. |
| **top-1 agreement** | The share of scored positions where the arm's highest-probability id matches the reference's. |
| **PPL** | Teacher-forced perplexity over the scored positions, `exp(mean NLL per position)`. |
| **dNLL** | Mean change in per-token negative log-likelihood against the reference, in nats — identical to the log of the PPL ratio, `ln(PPL_arm) − ln(PPL_ref)`. Negative means the arm is *more* likely than the reference on the scored positions. |
| **floor** | What `compare` does with a reference id that is absent from the candidate's stored top-K: it spreads the candidate's own tail mass uniformly over the missing ids. That is an **assumption**, and its hit rate varies by arm, so it can flatter or penalise an arm. |
| **`q` floor hits** | The rate at which that assumption was invoked. If it's material, raise `--top-k` before trusting the comparison. |
| **floor-free** | `kld-paired-floorfree.py`: a paired estimator that sums only over reference ids present in *both* arms' supports, so every floored value is excluded from both sides. **It is not a KL** — the retained probability mass doesn't sum to 1, so its absolute level is meaningless and only the delta, the paired t-statistic and the win-rate are interpretable. Its deltas are therefore *smaller in magnitude* than differences of the mean-KLD column and are not interchangeable with them. |

## Serving

| term | meaning |
|---|---|
| **vLLM 0.29.0** | The serving engine every measured arm ran on (TP=2 on Ampere). Its version matters: it refuses mixed-scheme MoE and has specific `group_size` handling — both documented in the build log. |
| **TP2** | Tensor parallelism across 2 GPUs. |
| **fused units** | vLLM merges modules at load: the experts become `RoutedExperts` (queried as `gate_proj`/`up_proj`/`down_proj`), the attention projections become one stacked `qkv_proj`, and the dense MLP's gate+up become `w13`. **Every shard inside one fused unit must share a single quantization scheme** — this is the constraint that killed E12 and shaped Path 4. |
| **kernels** | As the serve logs resolve them for vLLM 0.29.0: W8A8 experts → `CompressedTensorsW8A8Int8MoEMethod` → `TritonExperts`; weight-only W8A16 linears → `CompressedTensorsWNA16` → `AllSpark`; weight-only WNA16 MoE → the Marlin path. |

## Hosts

Two roles, named consistently throughout the docs — no hostnames:

| term | meaning |
|---|---|
| **the build node** | The GPU host that quantizes, serves, and captures KLD. Holds the checkpoints and the captures. |
| **the client host** | Drives the speed and extraction runs over LAN. Never serves a model. |
