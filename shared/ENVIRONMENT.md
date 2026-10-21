# Shared environment

One venv per repo, at the repo root: `.venv/` (i.e. `../.venv` from this
directory) — do not create nested per-project venvs.

## Rebuild recipe (verified stack)

```bash
# run from the repo root — the venv lives at <repo>/.venv, not in shared/
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch
.venv/bin/pip install "transformers==5.15.1" "llmcompressor==0.13.0" \
  "compressed-tensors==0.18.0" "accelerate>=0.33" "safetensors>=0.4" \
  "numpy>=1.26" "tokenizers>=0.21" "datasets>=4.0,<5" "jinja2>=3.1,<4" \
  "pytest>=8.0" "httpx" "ddgs" "lxml"
# llmcompressor 0.13.0 pins transformers<5.15 and drags in 5.14.1, whose
# save_pretrained is broken for offloaded models (Freaksterz Dockerfile).
.venv/bin/pip install --no-deps --force-reinstall "transformers==5.15.1"
```

Verify: `import torch, transformers, llmcompressor, compressed_tensors`
→ torch 2.13.0+cu130, transformers 5.15.1, llmcompressor 0.13.0, ct 0.18.0.

Notes: GPU here is host-passthrough 5090 (never install NVIDIA drivers in
WSL). CUDA entry points need `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
before importing torch.
