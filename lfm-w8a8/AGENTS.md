# LFM2.5-24B-A2B W8A8 — agent context

RTN-first workspace for **LFM2.5-24B-A2B** (Liquid AI, LFM2.5 MoE: gated
short convolutions + GQA + 64-expert sparse MoE). No GPTQ anywhere — design
choice: GPTQ prioritizes by calib activations and overfits to the corpus; RTN
is uniform. SmoothQuant folds are exact math, not fitting, so they're
allowed.

Standalone workspace for LFM2.5-24B-A2B. Several module names look
interchangeable with other MoE families but fail *silently* if copied over
(see "Do not").

## Do not

- Quantize the router (`feed_forward.gate`), the short-conv block, the
  embeddings, or the norms. Note the router is **not** `mlp.gate` here:
  LFM2 names the MoE container `feed_forward`, so the `mlp.gate` pattern
  other MoE families use (`re:.*mlp[.]gate$`) matches nothing and would
  silently quantize the router.
- Copy ignore tables across model families verbatim. The conv pattern here
  is `conv.*` (other families use names like `linear_attn.*`), and a foreign
  pattern matches nothing in this model.
- Assume zero-centered norms. `Lfm2MoeRMSNorm` is a **plain** RMSNorm
  (`w * x`), not the `(1 + w) * x` form other families' folds assume. A
  zero-centered fold here is silently wrong.
- Use `AutoModel` / `AutoModelForCausalLM`. It demotes `config.json` to the
  text-only config and vLLM rejects the checkpoint. Use the concrete
  `Lfm2MoeForCausalLM`, and let a failed import fail hard.
- Downcast leftovers to FP16. Norms, `expert_bias`, and non-Linears stay
  BF16 (training dtype — better range for folded gains). Scales stay FP32 as
  the quantizer emits them — never cast by hand.
- Expect fused expert tensors. LFM2 ships experts as **per-expert 2D
  `nn.Linear`s with a `.weight` suffix** (`feed_forward.experts.{e}.w1/w3/w2`).
  That is the opposite of the fused layout other MoE families ship
  (`mlp.experts.gate_up_proj` as one `nn.Parameter` with no suffix).
  Consequence: there is no fused-half arithmetic anywhere, and the
  fused-halves C-fold bug is structurally impossible.
- Add a `linear_attn` / GDN fold. This model has no GDN; the 30 non-attention
  layers are short convolutions and stay BF16.
- Fold a norm without scaling EVERY consumer of its output. `ffn_norm` feeds
  the router as well as all 64 experts, so fold A multiplies the gate columns
  by `s_a` too, or the router sees `x / s_a` (the 2026-10-16/17 SQ-W8A8
  regression: top-4 expert set changed for 62% of tokens). `operator_norm`
  feeds q, k AND v, so fold B scales all three (a v-only B is 21% off).
  `tests/test_fold_plan.py` pins both.
- Build the weight-only probe from the stock `W8A16` preset. In
  compressed-tensors 0.18 that preset is group-128, finer than Path 1's
  per-channel weights, so it flatters the weights. Use `--scheme
  W8A16-CHANNEL`, and never point a probe `--dst` at the shipped W8A8
  directory (the script refuses it).
- Use the shared parent env (`../.venv`, see `../shared/ENVIRONMENT.md`).
  Never create a nested `.venv` in this workspace.

## Do

- RTN: `QuantizationModifier(scheme="W8A8", targets=["Linear"],
  ignore=IGNORE_RTN)` from `scripts/lfm2_common.py`, accelerate-split load
  (`device_map="auto"`, per-GPU + 40GiB CPU caps, disk offload) — the 48GB
  source does NOT fit in the build node's RAM.
- Run `preflight_geometry()` before loading. It asserts the source really is
  LFM2.5-24B-A2B (40 layers, attn on 2/6/…/38, 38 MoE layers, 64 experts,
  no `shared_expert`/`linear_attn`/`mtp`/`visual`) and aborts on a mismatch.
- Keep `_force_moe_linearize_cpu()` in both quantize scripts: llmcompressor
  inherits each module's execution device as its conversion offload target,
  which piles ~46GB onto 2x24GB VRAM and OOMs. CPU-pinned conversion changes
  placement only, never values.
- Trust the census, not the vibe. RTN expects 7342 quantized Linears
  (64x38x3 experts + 10x4 attn + 2x3 dense); E12 expects the same total split
  4908 W8A8 / 2434 W8A16; Path 4 (`W8A8-EXPERTS`) expects 7296 W8A8 experts +
  46 W8A16 (40 attn + 6 dense) with 0 mode mismatches. A mismatch means do not
  ship the build.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` *before* importing
  torch in any CUDA entry point.
- Stop the serving job first. Quant builds need the ~48GB of VRAM actually
  free.

## Entry points

Model paths auto-resolve per host (`$LFM2_MODELS`, else `~/models`) — pass
`--src/--dst/--model/--out` only to override.

- RTN quantize (Path 1, READY): `python scripts/quantize_lfm2_rtn.py [--dry-run]`
- Weight-only isolation probe (Path 3, READY; vLLM 0.29.0 needs its
  `group_size: -1` fix, which the script writes):
  `python scripts/quantize_lfm2_rtn.py --scheme W8A16-CHANNEL [--dry-run]`
- Selective experts-only W8A8 split (Path 4, READY; 7296 experts W8A8 + 46
  attn/dense W8A16):
  `python scripts/quantize_lfm2_rtn.py --scheme W8A8-EXPERTS [--dry-run]`
- Capture (Path 2): `python scripts/capture-lfm2.py [--dry-run]`
- Alpha (Path 2; scale search is defective, see post-mortem):
  `python scripts/lfm2_alpha.py [--dry-run]`
- Smoothing (Path 2; `--folds` takes any subset of ABCE):
  `python scripts/apply-smoothing-lfm2.py [--dry-run]`
- E12 applier (Path 2; builds, but vLLM 0.29.0 cannot serve it):
  `python scripts/quantize_lfm2_sq.py [--dry-run]`
- Smoke: `python scripts/smoke.py --model <quantized-dir>`
- Tests: `../.venv/bin/pytest tests -q` (includes the fold wiring tests)

Path 2 ran end to end 2026-10-16/17 (seven first-contact bugs fixed; see the
build log in docs/accuracy-suggestions.md). E12 builds but vLLM 0.29.0 cannot
serve mixed-scheme MoE; smoothed-uniform (SQ-W8A8) served but regressed vs
Path 1 (KLD 0.70 vs 0.22). Diagnosed 2026-10-17: fold A scaled `ffn_norm`
without scaling the router that reads the same tensor. Fixed in
`apply-smoothing-lfm2.py` (not rebuilt), along with fold C having been
de-nested to expert 63 only. B and E are exact: B folds q, k AND v, so older
"v-only B" notes are wrong. Still open: the `lfm2_alpha.py` scale search is
defective (worse than plain RTN on held-out rows), and the smoothed/SQ
checkpoints were deleted. Measurements and next steps:
docs/accuracy-suggestions.md § "Path 2 post-mortem". Path 3 (the 2026-10-18
W8A16-CHANNEL probe: Path 1 weights, no activation quantizer) scored KLD
0.137 / top-1 84.1% / PPL 29.04 and ~+1% decode vs FP8, beating FP8 on every
metric: the whole gap to FP8 is activation quantization (0.079 KLD / 4.3 pt
top-1), and INT8 weights alone are better than FP8's. Path 4 (`--scheme
W8A8-EXPERTS`, built 2026-10-18) moves the 7296 MoE experts onto W8A8 while
keeping attention and dense layers 0-1 at W8A16: KLD 0.153 / top-1 82.9% / PPL
28.88 and +8.2..8.6% decode vs FP8 (recovering 80% of Path 1's activation KLD
cost at full W8A8 speed). Do not fire a heavy stage without a nod.
