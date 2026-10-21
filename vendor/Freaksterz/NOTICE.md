# vendored upstream reference — Freaksterz

Pristine copies from
[`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8)
(`recipe/`, fetched 2026-10-19), published under the Apache License 2.0
(`LICENSE` in this directory — the same terms as the repo root `LICENSE`).

## Files

- `kld-fullvocab.py` — teacher-forced full-vocab KLD vs an offline BF16
  reference. Pristine upstream copy.
- `kld-paired-floorfree.py` — floor-free paired comparison of two candidate
  captures against one shared reference. Pristine upstream copy.

## Modifications in the working copies

The pipeline runs the copies in `shared/benchmark-composes/scripts/`, not
these. Both working copies add a short provenance header (attribution travels
with the file); beyond that:

- `kld-fullvocab.py`: added a `ValueError` fallback around the model load —
  when the multimodal loader rejects a text-only MoE config (e.g.
  `Lfm2MoeConfig`), it retries with `AutoModelForCausalLM`. Apart from that
  change and the provenance header, the file is identical to this reference.
- `kld-paired-floorfree.py`: no functional change — identical to this
  reference apart from the provenance header.
