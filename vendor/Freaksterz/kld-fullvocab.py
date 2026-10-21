#!/usr/bin/env python3
"""kld-fullvocab.py — near-exact full-vocab KLD vs an oversized BF16 reference.

WHY THIS EXISTS (Muse W8A8 evaluation, 2026-08-11)
--------------------------------------------------
`eval/kld-compare.py` compares two SERVED models at top-20 truncation. That is
fine for ranking arms but has two gaps for Muse:

  1. The BF16 reference (51.9 GiB text stack) cannot be served normally on
     2x24 GB, and top-20 truncated KL is not comparable to published
     full-vocab numbers (Minachist: mean 0.00130 over 122,640 tokens).
  2. W8A8's ACTIVATION quantization exists only inside vLLM's INT8 kernels.
     Offline transformers scoring decompresses weights to BF16 and silently
     measures the weight-rounding error ONLY. A W8A8 KLD measured offline
     would be a systematic UNDERSTATEMENT.

METHOD
------
Reference capture is OFFLINE (transformers + accelerate offload, full vocab):
per position store the top-K (default 512) ids + exact log-softmax logprobs +
the exact tail log-mass log(1 - sum(topK p)). K=512 typically covers
>99.99% of the reference mass; the stored tail bound makes the truncation
error REPORTABLE instead of silent.

Candidate capture has two modes:
  * `capture-offline`  — same offline path. For weight-only quants (W8A16)
    this is EXACT end-to-end. For W8A8 it measures the weight component only
    (still useful: it isolates weight-rounding from activation-rounding).
  * `capture-vllm`     — teacher-forced through a served vLLM instance via
    /v1/completions prompt_logprobs, prompts sent as TOKEN IDS taken from the
    reference capture (identical positions by construction). This includes
    activation quant + real kernels. Serve with MAX_LOGPROBS >= --top-k.

`compare` computes KL(P||Q) over the reference's stored support:
    KL = sum_{i in topK(P)} p_i (log p_i - log q_i)   [+ reported tail bound]
q_i comes from the candidate's stored distribution; ids absent from the
candidate's support get the candidate's own tail mass spread uniformly
(reported as `q_floor_hits` — if that count is material, raise --top-k).
Also reports top-1 agreement and teacher-forced PPL per arm.

Both arms MUST share the tokenizer. Offline capture records the token ids;
vllm capture replays them; a length mismatch aborts.

USAGE (inside the muse-glimmer container or any torch+transformers>=5.15 env)
-----------------------------------------------------------------------------
  # 0) corpus: any plain-text file(s); scored as --n-seqs windows of --ctx.
  # 1) reference (BF16, offloaded; ~20-60 min for 240x512):
  python3 kld-fullvocab.py capture-offline --model /models-muse/Muse-Glimmer-30B-BF16 \
      --corpus wiki.test.raw --tag bf16 --max-gpu-gib 20 --out-dir /out
  # 2a) weight-only candidate offline:
  python3 kld-fullvocab.py capture-offline --model <W8A16-dir> --tag w8a16 \
      --corpus wiki.test.raw --ref /out/kldfv-bf16.npz --out-dir /out
  # 2b) end-to-end candidate through vLLM (server started with MAX_LOGPROBS=512):
  python3 kld-fullvocab.py capture-vllm --url http://localhost:11500 \
      --model muse-glimmer --tag w8a8 --ref /out/kldfv-bf16.npz --out-dir /out
  # 3) compare:
  python3 kld-fullvocab.py compare /out/kldfv-bf16.npz /out/kldfv-w8a8.npz
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- corpus
def build_windows(tok, corpus_paths, ctx, n_seqs, stride=None):
    text = "\n\n".join(Path(p).read_text(errors="replace") for p in corpus_paths)
    ids = tok(text, add_special_tokens=False).input_ids
    stride = stride or ctx
    wins = []
    for s in range(0, len(ids) - ctx, stride):
        wins.append(ids[s:s + ctx])
        if len(wins) >= n_seqs:
            break
    if len(wins) < n_seqs:
        print(f"WARNING: corpus only yields {len(wins)} windows of {ctx}", file=sys.stderr)
    return wins


# ------------------------------------------------------------ offline capture
def capture_offline(a):
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    wins = build_windows(tok, a.corpus, a.ctx, a.n_seqs)

    from transformers import AutoModelForCausalLM
    try:
        from transformers import AutoModelForMultimodalLM  # transformers>=5
        loader = AutoModelForMultimodalLM
    except ImportError:
        loader = AutoModelForCausalLM

    kw = dict(dtype="auto", trust_remote_code=True)
    if a.max_gpu_gib:
        import torch as _t
        mm = {i: f"{a.max_gpu_gib}GiB" for i in range(_t.cuda.device_count())}
        mm["cpu"] = f"{a.max_cpu_gib}GiB"
        kw.update(device_map="auto", max_memory=mm,
                  offload_folder=a.offload_dir, offload_state_dict=True)
    else:
        kw.update(device_map="auto")
    print(f"loading {a.model} ...", flush=True)
    model = loader.from_pretrained(a.model, **kw)
    model.eval()
    # Language decoder only: mirror --language-model-only. For Muse the text
    # path is reached via the top-level forward with no pixel inputs.

    K = a.top_k
    ids_out, top_ids, top_lps, tails, actual_lps = [], [], [], [], []
    t0 = time.time()
    for wi, w in enumerate(wins):
        inp = torch.tensor([w])
        with torch.no_grad():
            out = model(input_ids=inp.to(model.device), use_cache=False)
        logits = out.logits[0].float()                      # [ctx, vocab]
        lsm = torch.log_softmax(logits, dim=-1)
        # positions 0..ctx-2 predict tokens 1..ctx-1
        lsm = lsm[:-1]
        tgt = torch.tensor(w[1:])
        alp = lsm.gather(1, tgt.unsqueeze(1).to(lsm.device)).squeeze(1)
        tl, ti = torch.topk(lsm, K, dim=-1)
        # ⚠ The tail guard MUST be computed in float64. `1 - 1e-9` is NOT
        # representable in float32 (eps 1.19e-7) -> it rounds to exactly 1.0,
        # the clamp becomes a no-op, and any ultra-confident position whose
        # top-K mass rounds to 1.0 stores tail = log1p(-1) = -inf. Those -inf
        # tails then make compare() report mean KLD = inf for ANY candidate
        # captured this way (528 such positions existed in the Qwen3.8 BF16
        # REFERENCE itself; diagnosed 2026-08-15, see eval/diagnose-inf-tails.py).
        tail = torch.log1p(-torch.exp(tl.double()).sum(-1).clamp(max=1 - 1e-9))
        tail = tail.float()
        ids_out.append(np.array(w, dtype=np.int32))
        top_ids.append(ti.cpu().numpy().astype(np.int32))
        top_lps.append(tl.cpu().numpy().astype(np.float32))
        tails.append(tail.cpu().numpy().astype(np.float32))
        actual_lps.append(alp.cpu().numpy().astype(np.float32))
        if (wi + 1) % 10 == 0 or wi == len(wins) - 1:
            el = time.time() - t0
            print(f"  {wi + 1}/{len(wins)} windows  ({el:.0f}s, {el / (wi + 1):.1f}s/win)", flush=True)

    _save(a, ids_out, top_ids, top_lps, tails, actual_lps, mode="offline")
    return 0


# --------------------------------------------------------------- vLLM capture
def capture_vllm(a):
    import urllib.request

    K = a.top_k

    def post(path, payload):
        req = urllib.request.Request(a.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())

    if a.ref:
        ref = np.load(a.ref, allow_pickle=True)
        win_ids = ref["win_ids"]                            # [n, ctx] int32
    else:
        # Reference capture through vLLM itself: tokenize the corpus via the
        # server so both arms replay byte-identical token windows.
        assert a.corpus, "--corpus required when no --ref is given"
        text = "\n\n".join(Path(p).read_text(errors="replace") for p in a.corpus)
        ids = post("/tokenize", {"model": a.model, "prompt": text,
                                 "add_special_tokens": False})["tokens"]
        win_ids = [ids[s:s + a.ctx] for s in range(0, len(ids) - a.ctx, a.ctx)][:a.n_seqs]
        print(f"tokenized corpus: {len(ids)} tokens -> {len(win_ids)} windows of {a.ctx}")

    ids_out, top_ids, top_lps, tails, actual_lps = [], [], [], [], []
    t0 = time.time()
    for wi, w in enumerate(win_ids):
        w = [int(x) for x in w]
        resp = post("/v1/completions", {
            "model": a.model, "prompt": w, "max_tokens": 1,
            "temperature": 0, "prompt_logprobs": K})
        pl = resp["choices"][0]["prompt_logprobs"]
        assert len(pl) == len(w), f"window {wi}: {len(pl)} logprob rows != {len(w)} tokens"
        ti = np.full((len(w) - 1, K), -1, dtype=np.int32)
        tl = np.full((len(w) - 1, K), -100.0, dtype=np.float32)
        alp = np.full(len(w) - 1, np.nan, dtype=np.float32)
        tail = np.zeros(len(w) - 1, dtype=np.float32)
        for pos in range(1, len(w)):
            entry = pl[pos]
            if entry is None:
                continue
            items = sorted(((int(k), v["logprob"]) for k, v in entry.items()),
                           key=lambda kv: -kv[1])
            for j, (k, lp) in enumerate(items[:K]):
                ti[pos - 1, j] = k
                tl[pos - 1, j] = lp
            d = entry.get(str(w[pos]))
            if d is not None:
                alp[pos - 1] = d["logprob"]
            mass = sum(math.exp(lp) for _, lp in items[:K])
            # Python floats are float64, so 1-1e-9 IS representable here and
            # this guard genuinely fires (unlike the fp32 offline path above).
            tail[pos - 1] = math.log1p(-min(mass, 1 - 1e-9))
        ids_out.append(np.array(w, dtype=np.int32))
        top_ids.append(ti)
        top_lps.append(tl)
        tails.append(tail)
        actual_lps.append(alp)
        if (wi + 1) % 10 == 0 or wi == len(win_ids) - 1:
            el = time.time() - t0
            print(f"  {wi + 1}/{len(win_ids)} windows  ({el:.0f}s)", flush=True)

    _save(a, ids_out, top_ids, top_lps, tails, actual_lps, mode="vllm")
    return 0


def _save(a, ids_out, top_ids, top_lps, tails, actual_lps, mode):
    out = Path(a.out_dir) / f"kldfv-{a.tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, win_ids=np.stack(ids_out), top_ids=np.stack(top_ids),
        top_lps=np.stack(top_lps), tails=np.stack(tails),
        actual_lps=np.stack(actual_lps),
        meta=json.dumps({"tag": a.tag, "mode": mode, "top_k": a.top_k,
                         "model": getattr(a, "model", None),
                         "ts": time.strftime("%Y%m%d-%H%M%S")}))
    print(f"saved {out}")


# -------------------------------------------------------------------- compare
def compare(a):
    P = np.load(a.p, allow_pickle=True)
    Q = np.load(a.q, allow_pickle=True)
    mp, mq = json.loads(str(P["meta"])), json.loads(str(Q["meta"]))
    assert (P["win_ids"] == Q["win_ids"]).all(), "different corpora/tokenization"

    K = int(P["top_ids"].shape[-1])
    # Vocab size is MODEL-SPECIFIC and only affects the uniform floor assigned to
    # ids absent from the candidate's support (log(VOCAB-K) below). It was
    # hardcoded to Muse's 202048 during the 2026-08-11 W8A8 run; Qwen3.6 is
    # 248320. Mis-setting it biases only the floor term, so the error is bounded
    # by q_floor_hits -- check that number before trusting a comparison.
    VOCAB = int(a.vocab)

    Pid = P["top_ids"].astype(np.int32)
    Qid = Q["top_ids"].astype(np.int32)
    Plp = P["top_lps"].astype(np.float32)
    Qlp = Q["top_lps"].astype(np.float32)
    valid = (Pid[..., 0] >= 0) & (Qid[..., 0] >= 0)
    n = int(valid.sum())

    # sort Q's support per position; stable-merge against P's ids to align
    so = np.argsort(Qid, axis=-1)
    qs_ids = np.take_along_axis(Qid, so, -1)
    qs_lps = np.take_along_axis(Qlp, so, -1)
    qt_each = Q["tails"].astype(np.float32)[..., None] - math.log(VOCAB - K)
    # DEFENSIVE FLOOR: a capture produced with the pre-2026-08-15 fp32 tail
    # guard can contain -inf tails (the clamp was a no-op in float32). An -inf
    # tail turns every floor-hit id into an infinitely improbable event and
    # makes the reported mean KLD `inf` even when the candidate is perfect.
    # Floor at log(fp32 eps): if fp32 says the top-K mass is >= 1-eps, the true
    # tail mass is <= eps, so this is a conservative UPPER bound on divergence.
    n_inf_tail = int((~np.isfinite(qt_each)).sum())
    if n_inf_tail:
        qt_each = np.where(np.isfinite(qt_each),
                           qt_each,
                           math.log(float(np.finfo(np.float32).eps)) - math.log(VOCAB - K))
    qlp = np.broadcast_to(qt_each, Plp.shape).copy()
    matched = 0
    for w in range(Pid.shape[0]):
        ids = np.concatenate([qs_ids[w], Pid[w]], axis=-1)
        order = np.argsort(ids, axis=-1, kind="stable")
        ids_s = np.take_along_axis(ids, order, -1)
        dup = ids_s[:, :-1] == ids_s[:, 1:]
        keep = dup & (order[:, :-1] < K) & (order[:, 1:] >= K)
        rows, spos = np.nonzero(keep)
        if rows.size:
            qlp[w, rows, order[rows, spos + 1] - K] = qs_lps[w][rows, order[rows, spos]]
            matched += int(rows.size)

    kl = np.maximum(np.sum(np.exp(Plp) * (Plp - qlp), axis=-1), 0.0)[valid]
    agrees = int((Pid[..., 0] == Qid[..., 0])[valid].sum())
    floor_hits = n * K - matched
    pa = P["actual_lps"].astype(np.float64)[valid]
    qa = Q["actual_lps"].astype(np.float64)[valid]
    m = ~(np.isnan(pa) | np.isnan(qa))
    ppl_p, ppl_q = math.exp(-pa[m].mean()), math.exp(-qa[m].mean())

    print(f"P={mp['tag']} ({mp['mode']})  Q={mq['tag']} ({mq['mode']})  "
          f"K={mp['top_k']}/{mq['top_k']}  positions={n}")
    print(f"  mean KLD        : {kl.mean():.5f}  (± {kl.std() / math.sqrt(n):.6f})")
    print(f"  median KLD      : {np.median(kl):.6f}")
    print(f"  P90/P95/P99     : {np.percentile(kl, 90):.5f} / "
          f"{np.percentile(kl, 95):.5f} / {np.percentile(kl, 99):.5f}")
    print(f"  P99.9 / max     : {np.percentile(kl, 99.9):.4f} / {kl.max():.4f}")
    print(f"  top-1 agreement : {agrees / n * 100:.2f}%")
    print(f"  PPL P -> Q      : {ppl_p:.4f} -> {ppl_q:.4f}   ln ratio "
          f"{math.log(ppl_q) - math.log(ppl_p):.6f}")
    print(f"  ref tail mass   : mean {float(np.exp(P['tails']).mean()):.2e}  "
          f"(KL truncation bound; raise --top-k if not << mean KLD)")
    if n_inf_tail:
        print(f"  ⚠ candidate had {n_inf_tail} non-finite tail entries "
              f"(stale fp32-guard capture) — floored at log(fp32 eps); "
              f"KLD above is an UPPER BOUND. Re-capture to remove the bound.")
    print(f"  q floor hits    : {floor_hits} / {n * K} "
          f"({floor_hits / (n * K) * 100:.3f}% of gathered q values)")
    if mq["mode"] == "offline" and "w8a8" in mq["tag"].lower():
        print("  ⚠ candidate captured OFFLINE: activation quantization is NOT "
              "included in this number.")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture-offline")
    c.add_argument("--model", required=True)
    c.add_argument("--corpus", nargs="+", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--ctx", type=int, default=512)
    c.add_argument("--n-seqs", type=int, default=240)
    c.add_argument("--top-k", type=int, default=512)
    c.add_argument("--max-gpu-gib", type=int, default=0,
                   help="per-GPU cap; enables accelerate offload when set")
    c.add_argument("--max-cpu-gib", type=int, default=10)
    c.add_argument("--offload-dir", default="/tmp/kld-offload")
    c.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    c.set_defaults(func=capture_offline)

    v = sub.add_parser("capture-vllm")
    v.add_argument("--url", required=True)
    v.add_argument("--model", required=True)
    v.add_argument("--tag", required=True)
    v.add_argument("--ref", help="reference npz (token ids replayed); omit for "
                   "the reference capture itself (needs --corpus)")
    v.add_argument("--corpus", nargs="+", help="raw text file(s), reference capture only")
    v.add_argument("--ctx", type=int, default=512)
    v.add_argument("--n-seqs", type=int, default=240)
    v.add_argument("--top-k", type=int, default=512,
                   help="server must run with MAX_LOGPROBS >= this")
    v.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    v.set_defaults(func=capture_vllm)

    d = sub.add_parser("compare")
    d.add_argument("p", help="reference npz")
    d.add_argument("q", help="candidate npz")
    d.add_argument("--vocab", type=int, default=202048,
                   help="model vocab size; 202048=Muse, 248320=Qwen3.5/3.6/3.8. "
                        "Only affects the floor for unmatched ids (see q floor hits).")
    d.set_defaults(func=compare)

    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
