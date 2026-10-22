# quant-lab

Quantization pipelines for local inference — building, serving, and evaluating
low-precision checkpoints for open MoE models, plus the benchmark harness used
to measure them.

Each build is documented as a pipeline: the quantization recipe, the exact
module coverage, and a side-by-side evaluation against both the full-precision
source and the vendor's own reduced-precision checkpoint. The same two-leg
harness runs across every model and scheme — a teacher-forced full-vocab KLD
comparison for statistical parity, and long-context extraction with decode and
TTFT timing for speed and behaviour.

Coverage so far is INT8 (`compressed-tensors`, served via vLLM): W8A8
weight+activation, W8A16 weight-only, and component-precision splits that place
different schemes on different module classes. A SmoothQuant path was built and
evaluated, and is retained as a documented negative result. Further formats are
added here as they are tested on the same harness.

Terms used below — Path 1-4, `W8A8`/`W8A16`, and **floor-free** (the paired
estimator that removes the top-K tail-truncation floor) — are defined in
[`lfm-w8a8/docs/GLOSSARY.md`](lfm-w8a8/docs/GLOSSARY.md).

## LFM2.5-24B-A2B — first model

Liquid AI's 24B MoE (2B active/token; 64 routed experts, top-4; gated short
convolutions + GQA). Quantized from BF16 (47.94 GB) to ~24.8 GB — the same
footprint as the vendor FP8 checkpoint, but executing real INT8 tensor cores
rather than running FP8 weight-only.

**[Shipped build (`W8A8-EXPERTS`)](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS):** the 7,296 routed expert Linears (95.7% of
parameters) at W8A8 INT8, with attention projections and the layer 0–1 dense
MLPs kept at per-channel W8A16. It beats the vendor FP8 checkpoint on every
accuracy metric while running 8.2–8.6% faster decode on Ampere.

### Accuracy — teacher-forced full-vocab KLD vs BF16

240 × 512-token WikiText-2 windows (122,640 positions), top-K 512. Candidates
served through vLLM 0.29.0, TP2 on Ampere (2× RTX 3090 Ti / 3090).

| Checkpoint | Precision split (7,342 quantized Linears) | Mean KLD ↓ | Median KLD ↓ | Paired Δ vs FP8 (floor-free) | Top-1 agree ↑ | PPL ↓ (BF16 29.32) | `q` floor hits |
|---|---|---:|---:|---:|---:|---:|---:|
| **Path 3: `W8A16-CHANNEL`** | 7,342 W8A16 (per-channel INT8 weight-only) | **0.1369** | **0.0652** | −0.0364 (t = -57.9) | **84.05%** | **29.04** | 15.63% |
| **Path 4: [`W8A8-EXPERTS`](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS)** (shipped) | **7,296 experts W8A8 + 46 attn/dense W8A16** | **0.1527** | **0.0749** | **−0.0236 (t = -36.0)** | **82.91%** | **28.88** | 16.58% |
| [`LiquidAI/LFM2.5-24B-A2B-FP8`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-FP8) | Official FP8 (weight-only W8A16 on Ampere) | 0.1818 | 0.0954 | baseline | 81.18% | 29.29 | 18.27% |
| **Path 1: `W8A8`** | 7,342 W8A8 (experts + attn + dense 0–1) | 0.2161 | 0.1180 | +0.0265 (t = +36.8) | 79.72% | 29.21 | 19.83% |
| Path 2: `SQ-W8A8` (pre-fix, retired) | SmoothQuant + uniform W8A8 (unscaled router bug) | 0.7042 | — | — | 65.10% | 36.72 | — |

- **Paired Path 4 (`W8A8-EXPERTS`) vs official FP8:** Δ = -0.0236 ± 0.00066 (t = -36.0, −16.7% floor-free KLD), Path 4 closer to BF16 at **60.99%** of token positions.
- **Paired Path 4 (`W8A8-EXPERTS`) vs Path 1 (`W8A8`):** Δ = -0.0509 ± 0.00071 (t = -71.9, −30.3% floor-free KLD), Path 4 closer to BF16 at **68.46%** of token positions.

### Speed — sequential single-stream extraction

Measured from a separate client host over LAN (`extract_test.py`, sequential single-stream, `temperature=0`, `--enable-thinking`) across four real arXiv HTML papers spanning **5.5K to 50.6K prompt tokens**. Cells are decode tok/s / TTFT (paper IDs are arXiv IDs).

| Checkpoint | `2607.15866v1` (5,525 tok) | `2606.16140v1` (20,888 tok) | `2603.28052v1` (24,105 tok) | `2610.22111v1` (50,572 tok) | vs FP8 decode |
|---|---|---:|---:|---:|---:|
| `W8A16-CHANNEL` | 185.9 tok/s / 757 ms | 176.5 tok/s / 2,820 ms | 174.1 tok/s / 3,163 ms | 164.6 tok/s / 7,090 ms | ~+0.7% |
| **[`W8A8-EXPERTS`](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS)** (shipped) | **200.4 tok/s / 661 ms** | **189.9 tok/s / 2,919 ms** | **187.8 tok/s / 2,805 ms** | **176.9 tok/s / 6,322 ms** | **+8.2 to +8.6%** |
| LiquidAI FP8 (vendor) | 184.5 tok/s / 778 ms | 175.0 tok/s / 2,923 ms | 173.4 tok/s / 3,281 ms | 163.5 tok/s / 7,335 ms | baseline |
| `W8A8` | 197.2 tok/s / 653 ms | 187.4 tok/s / 2,884 ms | 185.1 tok/s / 2,752 ms | 174.5 tok/s / 6,234 ms | ~+6.9% |

*(Note: Path 3 vs Path 4 is a back-to-back comparison showing the pure +7.5% to +7.9% decode and −10.8% 50.6K-prefill gain from moving the 7,296 experts onto W8A8 INT8 tensor cores.)*

Methodology, per-path post-mortems, and the mixed-precision config live in
[`lfm-w8a8/README.md`](lfm-w8a8/README.md). The canonical copy of these results is the
[model card on Hugging Face](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS).

## Layout

```text
lfm-w8a8/                  LFM2.5-24B-A2B pipelines, unit tests, docs
  scripts/                 quantize / capture / smoothing / smoke
  tests/                   geometry + recipe unit tests (no model load)
  docs/                    recipe contract, post-mortems
  eval/                    vLLM serve smoke + serve composes
shared/                    venv recipe, eval harness (KLD/extract scripts + corpus)
vendor/Freaksterz/         pristine upstream KLD harness + license (see NOTICE)
```

Per-model directories are siblings under the repo root as more models are
added.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# then the documented final step in shared/ENVIRONMENT.md
# (transformers force-reinstall — install order matters)
.venv/bin/python -m pytest lfm-w8a8/tests -q
```

`shared/ENVIRONMENT.md` is the full environment recipe;
`lfm-w8a8/README.md` is the pipeline manual; `lfm-w8a8/docs/recipe.md`
is the contract — read it and `lfm-w8a8/AGENTS.md` before changing recipes.

## License

Pipeline code: Apache License 2.0 (`LICENSE`). Quantized checkpoints stay
under their base-model license — see the [model card on Hugging Face](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS).
Methodology and KLD harness adapted from
[`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8)
(Apache-2.0; pristine copies in `vendor/Freaksterz/`).
