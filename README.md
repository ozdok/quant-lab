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

| Checkpoint | Scheme | Mean KLD vs BF16 ↓ | Top-1 agree ↑ | PPL ↓ (BF16 29.32) |
|---|---|---:|---:|---:|
| `W8A16-CHANNEL` | 7,342 Linears, weight-only W8A16 | 0.1369 | 84.05% | 29.04 |
| **[`W8A8-EXPERTS`](https://huggingface.co/Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS)** (shipped) | 7,296 experts W8A8 + 46 attn/dense W8A16 | **0.1527** | **82.91%** | **28.88** |
| LiquidAI FP8 (vendor) | FP8 weight-only on SM86 | 0.1818 | 81.18% | 29.29 |
| `W8A8` | 7,342 Linears, uniform W8A8 | 0.2161 | 79.72% | 29.21 |

### Speed — sequential single-stream extraction

Four real arXiv papers, 5.5K–50.6K prompt tokens, temperature 0, measured over
LAN from a separate client host. Cells are decode tok/s / TTFT.

| Prompt (tokens) | Vendor FP8 | `W8A8` | `W8A16-CHANNEL` | **`W8A8-EXPERTS`** | vs FP8 |
|---|---:|---:|---:|---:|---:|
| 5,525 | 184.5 / 778 ms | 197.2 / 653 ms | 185.9 / 757 ms | **200.4 / 661 ms** | **+8.6%** |
| 20,888 | 175.0 / 2,923 ms | 187.4 / 2,884 ms | 176.5 / 2,820 ms | **189.9 / 2,919 ms** | **+8.5%** |
| 24,105 | 173.4 / 3,281 ms | 185.1 / 2,752 ms | 174.1 / 3,163 ms | **187.8 / 2,805 ms** | **+8.3%** |
| 50,572 | 163.5 / 7,335 ms | 174.5 / 6,234 ms | 164.6 / 7,090 ms | **176.9 / 6,322 ms** | **+8.2%** |

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
