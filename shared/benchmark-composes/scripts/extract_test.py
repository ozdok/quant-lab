#!/usr/bin/env python3
"""Offline extract-model harness: fetch a page, clean it, have a model format it.

Pipeline:
  1. DDGS.extract(url, fmt="text")  -> raw HTML
  2. clean_html(...)                -> stripped HTML
  3. POST {base}/chat/completions   -> model-formatted markdown

Defaults target a local OpenAI-compatible server at 127.0.0.1:8125.  --base-url http://<serve-host>:8051/v1  --base-url http://127.0.0.1:8125/v1

Dependencies come from the repo venv (shared/ENVIRONMENT.md recipe):
ddgs + httpx + lxml (lxml optional — regex fallback without it).

Examples:
    cd /path/to/extract-harness
    ./extract_test.py --save-dir ./out https://arxiv.org/html/2607.15866v1 --base-url http://<serve-host>:8051/v1
    ./extract_test.py --save-dir ./out https://arxiv.org/html/2606.16140v1 --base-url http://<serve-host>:8051/v1
    ./extract_test.py --save-dir ./out https://arxiv.org/html/2603.28052v1 --base-url http://<serve-host>:8051/v1
    ./extract_test.py --save-dir ./out https://arxiv.org/html/2610.22111v1 --base-url http://<serve-host>:8051/v1
  python scripts/extract_test.py https://example.com
  python scripts/extract_test.py https://example.com --model my-model
  python scripts/extract_test.py https://example.com --save-dir ./out
  python scripts/extract_test.py https://example.com --skip-clean
  python scripts/extract_test.py https://example.com --no-llm --save-dir ./out

Blank / omitted --model (or the placeholder "my-model") resolves to the first
id listed at {base}/models. Responses also print vLLM per-request metrics and
routed-expert summary when the server returns them.

Standalone notes (Oct 2026): this file carries its own copy of clean_html
(inlined below) so it stays single-file. Cleaning is currently
force-disabled in main() (cleaned = raw_html); --skip-clean/--no-svg/--no-base64
are parsed but inert until that block is re-enabled.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from ddgs import DDGS

import httpx


# --- clean_html, inlined so this harness stays single-file ---
# Kept verbatim so this harness stays single-file. Uses lxml when available;
# falls back to a regex path otherwise.
SCRIPT_PATTERN = r"<[ ]*script\b[^>]*>.*?<\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style\b[^>]*>.*?<\/[ ]*style[ ]*>"
NOSCRIPT_PATTERN = r"<[ ]*noscript\b[^>]*>.*?<\/[ ]*noscript[ ]*>"
META_PATTERN = r"<[ ]*meta\b[^>]*>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link\b[^>]*>"
BASE64_IMG_PATTERN = (
    r'<img\b[^>]*\bsrc\s*=\s*["\']data:image/[^;]+;base64,[^"\']+["\'][^>]*>'
)
SVG_PATTERN = r"<[ ]*svg\b[^>]*>.*?<\/[ ]*svg[ ]*>"

# Tags that are almost never useful for extract summaries.
_DROP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "meta",
        "link",
        "iframe",
        "object",
        "embed",
        "canvas",
        "dialog",
        "form",
        "button",
        "input",
        "select",
        "textarea",
        "label",
        "option",
        "svg",
        "path",
        "symbol",
        "use",
        "header",
        "footer",
        "nav",
        "aside",
    }
)

# Drop whole subtrees whose class/id looks like chrome (arxiv, docs sites, blogs).
_CHROME_ATTR_RE = re.compile(
    r"(?:^|[\s_-])"
    r"(?:toc|navbar|nav-?bar|sidebar|breadcrumb|cookie|consent|banner|"
    r"announcement|modal|popup|popover|toolbar|share|social|newsletter|"
    r"subscribe|login|signup|advert|ads?|promo|infobox|watermark|"
    r"html-header|page_navbar|ltx_page_navbar|ltx_TOC|ds-announcement)"
    r"(?:[\s_-]|$)",
    re.IGNORECASE,
)

# Keep only attributes that carry meaning for reading/tables/links.
_KEEP_ATTRS = frozenset({"href", "src", "alt", "colspan", "rowspan", "title"})

_WS_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_EMPTY_TAG_RE = re.compile(
    r"<([a-zA-Z][\w:-]*)(?:\s[^>]*)?>\s*</\1>",
    re.IGNORECASE,
)


def replace_svg(html: str, new_content: str = "") -> str:
    """Remove SVG bodies (default) or replace inner content."""
    if new_content:
        return re.sub(
            r"(<[ ]*svg\b[^>]*>)(.*?)(<\/[ ]*svg[ ]*>)",
            lambda m: f"{m.group(1)}{new_content}{m.group(3)}",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return re.sub(SVG_PATTERN, "", html, flags=re.IGNORECASE | re.DOTALL)


def replace_base64_images(html: str, new_image_src: str = "") -> str:
    """Drop base64 ``<img>`` tags, or rewrite src if ``new_image_src`` is set."""
    if new_image_src:
        return re.sub(
            BASE64_IMG_PATTERN,
            f'<img src="{new_image_src}"/>',
            html,
            flags=re.IGNORECASE,
        )
    return re.sub(BASE64_IMG_PATTERN, "", html, flags=re.IGNORECASE)


def _collapse_whitespace(html: str) -> str:
    html = html.replace("\r\n", "\n").replace("\r", "\n")
    html = _WS_RE.sub(" ", html)
    html = _BLANK_LINES_RE.sub("\n\n", html)
    return html.strip()


def _strip_empty_tags(html: str, passes: int = 3) -> str:
    for _ in range(passes):
        nxt = _EMPTY_TAG_RE.sub("", html)
        if nxt == html:
            break
        html = nxt
    return html


def _looks_like_chrome(classes: str, elem_id: str) -> bool:
    blob = f"{classes} {elem_id}".strip()
    return bool(blob) and bool(_CHROME_ATTR_RE.search(blob))


def _drop_elements(root, tags: Iterable[str]) -> None:
    for tag in tags:
        for el in root.xpath(f".//{tag}"):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def _clean_with_lxml(
    html: str,
    *,
    clean_svg: bool,
    clean_base64: bool,
    strip_chrome: bool,
    strip_attrs: bool,
) -> str:
    from lxml import html as lhtml

    # fragment_fromstring is brittle on full documents; document_fromstring
    # recovers well on real-world / arXiv HTML.
    doc = lhtml.document_fromstring(html)

    drop = set(_DROP_TAGS)
    if not clean_svg:
        drop -= {"svg", "path", "symbol", "use"}
    _drop_elements(doc, drop)

    # Always kill head chrome leftovers (title is kept via body/h1 usually).
    for el in doc.xpath("//head"):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

    if strip_chrome:
        # Class/id based chrome (TOC, banners, arxiv header remnants, etc.).
        for el in list(doc.iter()):
            if not hasattr(el, "get"):
                continue
            classes = el.get("class") or ""
            elem_id = el.get("id") or ""
            if _looks_like_chrome(classes, elem_id):
                parent = el.getparent()
                if parent is not None:
                    parent.remove(el)

    if clean_base64:
        for img in doc.xpath(".//img"):
            src = img.get("src") or ""
            if src.startswith("data:image/") and ";base64," in src:
                parent = img.getparent()
                if parent is not None:
                    parent.remove(img)

    if strip_attrs:
        for el in doc.iter():
            if not hasattr(el, "attrib"):
                continue
            for attr in list(el.attrib):
                # Keep cite link text useful; drop long title tooltips that
                # repeat paper titles thousands of times in arXiv HTML.
                if attr == "title" and el.tag in {"a", "cite"}:
                    del el.attrib[attr]
                    continue
                if attr not in _KEEP_ATTRS:
                    del el.attrib[attr]

    # Prefer body content if present.
    body = doc.find("body")
    target = body if body is not None else doc
    out = lhtml.tostring(target, encoding="unicode", method="html")
    out = _strip_empty_tags(out)
    return _collapse_whitespace(out)


def _clean_with_regex(
    html: str,
    *,
    clean_svg: bool,
    clean_base64: bool,
) -> str:
    flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
    html = re.sub(SCRIPT_PATTERN, "", html, flags=flags)
    html = re.sub(STYLE_PATTERN, "", html, flags=flags)
    html = re.sub(NOSCRIPT_PATTERN, "", html, flags=flags)
    html = re.sub(META_PATTERN, "", html, flags=flags)
    html = re.sub(COMMENT_PATTERN, "", html, flags=flags)
    html = re.sub(LINK_PATTERN, "", html, flags=flags)
    # Best-effort chrome drops without a DOM.
    for tag in ("header", "footer", "nav", "aside", "dialog", "form"):
        html = re.sub(
            rf"<[ ]*{tag}\b[^>]*>.*?<\/[ ]*{tag}[ ]*>",
            "",
            html,
            flags=flags,
        )
    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
    # Strip most attributes (keep href/src/alt).
    def _trim_tag(m: re.Match[str]) -> str:
        name = m.group(1)
        attrs = m.group(2) or ""
        kept: list[str] = []
        for am in re.finditer(
            r"""([^\s=]+)\s*=\s*(["'])(.*?)\2""",
            attrs,
            flags=re.DOTALL,
        ):
            key = am.group(1).lower()
            if key in _KEEP_ATTRS and not (key == "title"):
                kept.append(f'{key}="{am.group(3)}"')
        if kept:
            return f"<{name} {' '.join(kept)}>"
        return f"<{name}>"

    html = re.sub(r"<([a-zA-Z][\w:-]*)(\s[^>]*)?>", _trim_tag, html)
    html = _strip_empty_tags(html)
    return _collapse_whitespace(html)


def clean_html(
    html: str,
    clean_svg: bool = True,
    clean_base64: bool = True,
    *,
    strip_chrome: bool = True,
    strip_attrs: bool = True,
) -> str:
    """Clean HTML for LLM consumption.

    Defaults are aggressive (svg/base64/chrome/attrs on) because the extract
    harness showed attribute + TOC chrome dominating the char budget.
    """
    if not html:
        return ""

    try:
        return _clean_with_lxml(
            html,
            clean_svg=clean_svg,
            clean_base64=clean_base64,
            strip_chrome=strip_chrome,
            strip_attrs=strip_attrs,
        )
    except Exception:
        return _clean_with_regex(
            html,
            clean_svg=clean_svg,
            clean_base64=clean_base64,
        )

_SYSTEM_PROMPT = """You are an expert content analyst. Your job is to process web content and create a comprehensive yet concise summary that preserves all important information while dramatically reducing bulk.

Create a well-structured markdown summary that includes:
1. Key excerpts (quotes, code snippets, important facts) in their original format
2. Comprehensive summary of all other important information
3. Proper markdown formatting with headers, bullets, and emphasis

Your goal is to preserve ALL important information while reducing length. Never lose key facts, figures, insights, or actionable information. Make it scannable and well-organized."""

slug = None  # global for save_dir / run_dir usage

def _build_user_prompt(url: str, body: str) -> str:
    return f"""Please process this web content and create a comprehensive markdown summary:

Source: {url}

CONTENT TO PROCESS:
{body}

Create a markdown summary that captures all key information in a well-organized, scannable format. Include important quotes and code snippets in their original formatting. Focus on actionable information, specific details, and unique insights."""


def _slug_from_url(url: str) -> str:
    host = urlparse(url).netloc.replace(":", "_") or "page"
    path = urlparse(url).path.strip("/").replace("/", "_")[:40]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [stamp, host]
    if path:
        parts.append(path)
    return "_".join(parts)


def _fetch_html(url: str, *, timeout: float) -> str:
    with DDGS(timeout=timeout) as client:
        result = client.extract(url, fmt="text_rich")
    content = result.get("content", "")
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return str(content)


def _auth_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _resolve_model(base_url: str, api_key: str, model: str, *, timeout: float) -> str:
    """Use endpoint default when model is blank or the 'my-model' placeholder."""
    name = (model or "").strip()
    if name and name.lower() not in {"my-model", "default", "auto"}:
        return name

    endpoint = base_url.rstrip("/") + "/models"
    with httpx.Client(timeout=timeout) as http:
        resp = http.get(endpoint, headers=_auth_headers(api_key))
        resp.raise_for_status()
        data = resp.json()

    ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
    if not ids:
        raise RuntimeError(f"No models listed at {endpoint}")
    chosen = str(ids[0])
    if name:
        print(f"      --model {name!r} -> endpoint default {chosen!r}")
    else:
        print(f"      --model omitted -> endpoint default {chosen!r}")
    if len(ids) > 1:
        print(f"      available: {', '.join(ids)}")
    return chosen


def _chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    enable_thinking: bool = False,
) -> tuple[str, dict[str, Any]]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # vLLM: return per-request timing when server has --enable-per-request-metrics
        "include_metrics": True,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }

    with httpx.Client(timeout=timeout) as http:
        resp = http.post(endpoint, headers=_auth_headers(api_key), json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat/completions response shape: {data!r}") from exc
    return str(text), data


# Minimal .npy dtype → (array.array typecode, itemsize). Covers what vLLM emits
# for routed_experts (typically |u1 / <i4 / <i8). No numpy required.
_NPY_DTYPES: dict[str, tuple[str, int]] = {
    "|u1": ("B", 1),
    "|i1": ("b", 1),
    "<u2": ("H", 2),
    "<i2": ("h", 2),
    "<u4": ("I", 4),
    "<i4": ("i", 4),
    "<u8": ("Q", 8),
    "<i8": ("q", 8),
    "<f4": ("f", 4),
    "<f8": ("d", 8),
    ">u2": ("H", 2),
    ">i2": ("h", 2),
    ">u4": ("I", 4),
    ">i4": ("i", 4),
    ">u8": ("Q", 8),
    ">i8": ("q", 8),
    ">f4": ("f", 4),
    ">f8": ("d", 8),
}


def _load_npy(raw: bytes) -> tuple[tuple[int, ...], str, list[Any]]:
    """Parse a NumPy .npy v1/v2 buffer with stdlib only (no pickle)."""
    import ast
    import array
    import struct

    if not raw.startswith(b"\x93NUMPY"):
        raise ValueError("not a .npy file (missing magic)")
    major, minor = raw[6], raw[7]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        header_start = 10
    elif major == 2:
        header_len = struct.unpack("<I", raw[8:12])[0]
        header_start = 12
    else:
        raise ValueError(f"unsupported .npy version {major}.{minor}")

    header = raw[header_start : header_start + header_len]
    meta = ast.literal_eval(header.decode("latin1").strip())
    if meta.get("fortran_order"):
        raise ValueError("fortran_order .npy not supported")
    descr = str(meta["descr"])
    shape = tuple(int(x) for x in meta["shape"])
    if descr not in _NPY_DTYPES:
        raise ValueError(f"unsupported .npy dtype: {descr}")

    typecode, itemsize = _NPY_DTYPES[descr]
    data = raw[header_start + header_len :]
    n_items = 1
    for dim in shape:
        n_items *= dim
    expected = n_items * itemsize
    if len(data) < expected:
        raise ValueError(f".npy truncated: need {expected} bytes, got {len(data)}")

    arr = array.array(typecode)
    arr.frombytes(data[:expected])
    if descr.startswith(">") and itemsize > 1:
        arr.byteswap()
    return shape, descr, arr.tolist()


def _flatten_nested(obj: Any) -> list[Any]:
    """Flatten nested lists into a 1-d Python list; return shape via caller."""
    if not isinstance(obj, list):
        return [obj]
    out: list[Any] = []
    for item in obj:
        if isinstance(item, list):
            out.extend(_flatten_nested(item))
        else:
            out.append(item)
    return out


def _nested_shape(obj: Any) -> list[int]:
    shape: list[int] = []
    cur: Any = obj
    while isinstance(cur, list):
        shape.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return shape


def _decode_routed_experts(blob: Any) -> dict[str, Any] | None:
    """Decode choices[].routed_experts (base64 NPY or nested lists). Stdlib only."""
    if blob is None:
        return None

    if isinstance(blob, list):
        shape = _nested_shape(blob)
        flat = _flatten_nested(blob)
        dtype = "list"
    elif isinstance(blob, str):
        try:
            shape_t, dtype, flat = _load_npy(base64.b64decode(blob))
            shape = list(shape_t)
        except Exception as exc:  # noqa: BLE001
            return {
                "note": f"npy decode failed: {exc}; raw base64 length={len(blob)}"
            }
    else:
        return {"note": f"unsupported routed_experts type: {type(blob).__name__}"}

    info: dict[str, Any] = {
        "shape": shape,
        "dtype": dtype,
    }
    if len(shape) == 3 and flat:
        tokens, layers, top_k = (int(x) for x in shape)
        ints = [int(x) for x in flat]
        info.update(
            {
                "tokens": tokens,
                "layers": layers,
                "top_k": top_k,
                "unique_experts": len(set(ints)),
                "expert_id_min": min(ints),
                "expert_id_max": max(ints),
            }
        )
        from collections import Counter

        info["hot_experts_top5"] = Counter(ints).most_common(5)
    return info


def _f(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _derive_speed_metrics(
    usage: dict[str, Any],
    metrics: dict[str, Any] | None,
    *,
    wall_s: float,
) -> dict[str, Any]:
    """Turn vLLM's awkward timing fields into useful rates.

    vLLM definitions (per their docs):
      queue_time_ms          — wait in scheduler queue before processing
      time_to_first_token_ms — scheduled → first output token (≈ prefill + first decode)
      generation_time_ms     — first output token → last (decode only; excludes queue+TTFT)
      mean_itl_ms            — mean gap between successive output tokens (decode)
      tokens_per_second      — completion_tokens / (TTFT+gen)  [includes prefill!]

    Derived:
      prefill_tps   ≈ prompt_tokens / (TTFT_s)          # prefill proxy
      decode_tps    ≈ (completion_tokens - 1) / gen_s   # pure decode (matches ITL)
      decode_tps_itl ≈ 1000 / mean_itl_ms               # from mean ITL
      e2e_tps       ≈ completion_tokens / (TTFT+gen)_s  # same basis as vLLM's TPS
      server_total_ms = queue + TTFT + generation
    """
    prompt_tok = usage.get("prompt_tokens")
    completion_tok = usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    m = metrics or {}

    ttft_ms = _f(m.get("time_to_first_token_ms"))
    gen_ms = _f(m.get("generation_time_ms"))
    queue_ms = _f(m.get("queue_time_ms"))
    itl_ms = _f(m.get("mean_itl_ms"))
    vllm_tps = _f(m.get("tokens_per_second"))

    out: dict[str, Any] = {
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "cached_tokens": cached,
        "wall_s": round(wall_s, 4),
        "queue_ms": queue_ms,
        "ttft_ms": ttft_ms,
        "generation_ms": gen_ms,
        "mean_itl_ms": itl_ms,
        "vllm_tokens_per_second": vllm_tps,  # includes prefill; often misleading
    }

    if wall_s > 0 and completion_tok:
        out["client_tps"] = completion_tok / wall_s

    if ttft_ms is not None and gen_ms is not None:
        inference_ms = ttft_ms + gen_ms  # scheduled → last token
        out["inference_ms"] = inference_ms
        if queue_ms is not None:
            out["server_total_ms"] = queue_ms + inference_ms
        if completion_tok and inference_ms > 0:
            out["e2e_tps"] = completion_tok / (inference_ms / 1000.0)

    # Prefill proxy: prompt tokens processed during TTFT window
    if ttft_ms is not None and ttft_ms > 0 and prompt_tok:
        out["prefill_tps"] = prompt_tok / (ttft_ms / 1000.0)
        out["prefill_ms"] = ttft_ms  # alias; TTFT ≈ prefill-dominated

    # Pure decode: tokens after the first, over generation_time_ms
    if gen_ms is not None and gen_ms > 0 and completion_tok is not None:
        decode_tokens = max(completion_tok - 1, 0)
        out["decode_tokens"] = decode_tokens  # intervals / tokens after first
        if decode_tokens > 0:
            out["decode_tps"] = decode_tokens / (gen_ms / 1000.0)
        elif completion_tok == 1:
            out["decode_tps"] = None  # single token → no decode interval

    if itl_ms is not None and itl_ms > 0:
        out["decode_tps_from_itl"] = 1000.0 / itl_ms

    return out


def _print_llm_diagnostics(raw_resp: dict[str, Any], *, wall_s: float) -> dict[str, Any]:
    """Surface fields most clients do not display; return derived metrics for meta.json."""
    usage = raw_resp.get("usage") or {}
    metrics = raw_resp.get("metrics") if isinstance(raw_resp.get("metrics"), dict) else None
    derived = _derive_speed_metrics(usage, metrics, wall_s=wall_s)

    # --- tokens / client ---
    parts = [f"wall={wall_s:.2f}s"]
    if derived.get("prompt_tokens") is not None:
        parts.append(
            f"tok={derived['prompt_tokens']}+{derived.get('completion_tokens')}"
        )
    if derived.get("cached_tokens"):
        parts.append(f"cached={derived['cached_tokens']}")
    if derived.get("client_tps") is not None:
        parts.append(f"client_tps={derived['client_tps']:.1f}")
    print(f"      usage: {', '.join(parts)}")

    if not metrics:
        print(
            "      server metrics: (none — enable with vLLM --enable-per-request-metrics)"
        )
    else:
        # Raw vLLM fields (for cross-check)
        raw_bits = []
        if derived.get("queue_ms") is not None:
            raw_bits.append(f"queue={derived['queue_ms']:.2f}ms")
        if derived.get("ttft_ms") is not None:
            raw_bits.append(f"ttft={derived['ttft_ms']:.1f}ms")
        if derived.get("generation_ms") is not None:
            raw_bits.append(f"gen={derived['generation_ms']:.1f}ms")
        if derived.get("mean_itl_ms") is not None:
            raw_bits.append(f"itl={derived['mean_itl_ms']:.2f}ms")
        if derived.get("vllm_tokens_per_second") is not None:
            raw_bits.append(
                f"vllm_tps={derived['vllm_tokens_per_second']:.1f} (incl. prefill)"
            )
        print(f"      vllm raw: {', '.join(raw_bits)}")

        # Actual useful numbers
        useful = []
        if derived.get("prefill_tps") is not None:
            useful.append(
                f"prefill≈{derived['prefill_tps']:.1f} tok/s "
                f"({derived['prompt_tokens']} tok / {derived['ttft_ms']:.1f}ms TTFT)"
            )
        if derived.get("decode_tps") is not None:
            useful.append(
                f"decode={derived['decode_tps']:.1f} tok/s "
                f"({derived['decode_tokens']} tok after first / "
                f"{derived['generation_ms']:.1f}ms gen)"
            )
        elif derived.get("completion_tokens") == 1:
            useful.append("decode=n/a (single output token)")
        if derived.get("decode_tps_from_itl") is not None:
            useful.append(f"decode_from_itl={derived['decode_tps_from_itl']:.1f} tok/s")
        if derived.get("e2e_tps") is not None:
            useful.append(
                f"e2e={derived['e2e_tps']:.1f} tok/s "
                f"(out / TTFT+gen={derived['inference_ms']:.1f}ms)"
            )
        if derived.get("server_total_ms") is not None:
            useful.append(f"server_total={derived['server_total_ms']:.1f}ms (queue+TTFT+gen)")
        for line in useful:
            print(f"      speed: {line}")

    choice0 = (raw_resp.get("choices") or [{}])[0]
    experts = choice0.get("routed_experts")
    if experts is None:
        experts = raw_resp.get("prompt_routed_experts")
    decoded = _decode_routed_experts(experts)
    if decoded is None:
        print(
            "      routed_experts: (none — enable with vLLM --enable-return-routed-experts)"
        )
    elif "note" in decoded and "shape" not in decoded:
        print(f"      routed_experts: {decoded['note']}")
    else:
        hot = decoded.get("hot_experts_top5")
        hot_s = ""
        if hot:
            hot_s = " hot=" + ",".join(f"{eid}:{n}" for eid, n in hot)
        print(
            f"      routed_experts: shape={decoded.get('shape')} "
            f"unique={decoded.get('unique_experts')} "
            f"ids=[{decoded.get('expert_id_min')},{decoded.get('expert_id_max')}]"
            f"{hot_s}"
        )
        derived["routed_experts_summary"] = decoded

    return derived


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DDGS extract -> clean_html -> local chat/completions formatting harness",
    )
    p.add_argument("url", help="Page URL to fetch")
    p.add_argument(
        "--model",
        default="",
        help=(
            "Model id for /chat/completions. Blank, 'my-model', 'default', or 'auto' "
            "uses the first id from {base}/models"
        ),
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8125/v1",
        help="OpenAI-compatible API base (default: %(default)s)",
    )
    p.add_argument(
        "--api-key",
        default="",
        help="Optional Bearer token (or set EXTRACT_TEST_API_KEY)",
    )
    p.add_argument("--timeout", type=float, default=60.0, help="DDGS fetch timeout seconds")
    p.add_argument(
        "--llm-timeout",
        type=float,
        default=360.0,
        help="chat/completions HTTP timeout seconds",
    )
    p.add_argument(
        "--max-input-chars",
        type=int,
        default=200_000,
        help="Truncate cleaned HTML before sending to the model",
    )
    p.add_argument("--max-tokens", type=int, default=20000, help="max_tokens for the model")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Ask for <think> trace (vLLM reasoning parser puts it in "
        "message.reasoning, saved to thinking.md; content stays clean). "
        "NOTE: this sets chat_template_kwargs.enable_thinking=true, which is "
        "already the vLLM reasoning parser's default (vllm/parser/qwen3.py) "
        "and which models whose chat template lacks the variable (e.g. "
        "LFM2.5) ignore entirely — on those it is a no-op. The split is "
        "driven by the server's --reasoning-parser regardless.",
    )
    p.add_argument(
        "--skip-clean",
        action="store_true",
        help="Send raw HTML to the model (skip clean_html)",
    )
    p.add_argument(
        "--no-svg",
        action="store_true",
        help="Do not replace SVG bodies (clean_html clean_svg=False)",
    )
    p.add_argument(
        "--no-base64",
        action="store_true",
        help="Do not strip base64 images (clean_html clean_base64=False)",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Only fetch + clean; do not call the model",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Write raw.html / cleaned.html / formatted.md (+ meta.json) here",
    )
    p.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Override system prompt from a text file",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    api_key = args.api_key or os.environ.get("EXTRACT_TEST_API_KEY", "")

    model = args.model
    if not args.no_llm:
        try:
            model = _resolve_model(
                args.base_url,
                api_key,
                args.model,
                timeout=min(args.llm_timeout, 30.0),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error: could not resolve model: {exc}", file=sys.stderr)
            return 1

    print(f"[1/3] DDGS extract (fmt=text): {args.url}")
    t0 = time.perf_counter()
    try:
        raw_html = _fetch_html(args.url, timeout=args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"error: DDGS extract failed: {exc}", file=sys.stderr)
        return 1
    fetch_s = time.perf_counter() - t0
    print(f"      raw HTML: {len(raw_html):,} chars in {fetch_s:.1f}s")

    # if args.skip_clean:
    cleaned = raw_html
    print("[2/3] clean_html: skipped (--skip-clean)")
    # else:
    #     print("[2/3] clean_html")
    #     cleaned = clean_html(
    #         raw_html,
    #         clean_svg=not args.no_svg,
    #         clean_base64=not args.no_base64,
    #     )
    #     ratio = (100.0 * len(cleaned) / len(raw_html)) if raw_html else 0.0
    #     print(
    #         f"      cleaned: {len(cleaned):,} chars ({ratio:.1f}% of raw; "
    #         f"svg={'on' if not args.no_svg else 'off'}, "
    #         f"base64={'on' if not args.no_base64 else 'off'}, "
    #         f"chrome/attrs stripped)"
    #     )

    body_for_llm = cleaned
    truncated = False
    if len(body_for_llm) > args.max_input_chars:
        body_for_llm = body_for_llm[: args.max_input_chars]
        truncated = True
        print(f"      truncated for LLM: {len(cleaned):,} -> {len(body_for_llm):,} chars")

    save_dir = args.save_dir
    if save_dir is None and not args.no_llm:
        # still useful to dump next to cwd when comparing models
        pass
    if save_dir is not None:
        slug = _slug_from_url(args.url)
        run_dir = save_dir / slug
        _write(run_dir / "raw.html", raw_html)
        _write(run_dir / "cleaned.html", cleaned)
        print(f"      saved intermediates under {run_dir}")

    if args.no_llm:
        print("[3/3] LLM: skipped (--no-llm)")
        if save_dir is None:
            # print a short preview so the CLI isn't silent
            preview = cleaned[:2000]
            print("--- cleaned preview ---")
            print(preview)
            if len(cleaned) > 2000:
                print(f"... [{len(cleaned) - 2000:,} more chars]")
        return 0

    system = _SYSTEM_PROMPT
    if args.system_prompt_file:
        system = args.system_prompt_file.read_text(encoding="utf-8")

    user = _build_user_prompt(args.url, body_for_llm)
    print(f"[3/3] POST {args.base_url.rstrip('/')}/chat/completions  model={model!r}")
    t1 = time.perf_counter()
    try:
        formatted, raw_resp = _chat_completions(
            base_url=args.base_url,
            api_key=api_key,
            model=model,
            system=system,
            user=user,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.llm_timeout,
            enable_thinking=args.enable_thinking,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: chat/completions failed: {exc}", file=sys.stderr)
        return 1
    llm_s = time.perf_counter() - t1
    print(f"      model output: {len(formatted):,} chars in {llm_s:.1f}s")
    speed = _print_llm_diagnostics(raw_resp, wall_s=llm_s)

    print("\n========== FORMATTED ==========\n")
    print(formatted)
    print("\n================================\n")

    if save_dir is not None:
        # slug = _slug_from_url(args.url)
        run_dir = save_dir / slug
        _write(run_dir / "formatted.md", formatted)
        try:
            thinking = (
                raw_resp.get("choices", [{}])[0]
                .get("message", {})
                .get("reasoning")
            )
        except (AttributeError, IndexError, TypeError):
            thinking = None
        _write(run_dir / "thinking.md", (thinking or ""))
        usage = dict(raw_resp.get("usage") or {})
        try:
            details = dict(usage.get("completion_tokens_details") or {})
            reasoning = details.get("reasoning_tokens")
            total = usage.get("completion_tokens")
            if isinstance(reasoning, int) and isinstance(total, int):
                details["output_tokens"] = total - reasoning
                usage["completion_tokens_details"] = details
        except (AttributeError, TypeError, ValueError):
            pass
        meta = {
            "url": args.url,
            "model": model,
            "model_arg": args.model,
            "base_url": args.base_url,
            "raw_chars": len(raw_html),
            "cleaned_chars": len(cleaned),
            "sent_chars": len(body_for_llm),
            "truncated": truncated,
            "output_chars": len(formatted),
            "fetch_seconds": round(fetch_s, 3),
            "llm_seconds": round(llm_s, 3),
            "skip_clean": args.skip_clean,
            "clean_svg": not args.no_svg,
            "clean_base64": not args.no_base64,
            "usage": usage,
            "metrics": raw_resp.get("metrics"),
            "speed": speed,
        }
        _write(run_dir / "meta.json", json.dumps(meta, indent=2, default=str) + "\n")
        # Keep full response (may include large routed_experts base64).
        _write(run_dir / "response.json", json.dumps(raw_resp, indent=2) + "\n")
        print(f"saved: {run_dir / 'formatted.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
