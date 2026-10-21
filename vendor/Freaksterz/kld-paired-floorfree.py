#!/usr/bin/env python3
"""kld-paired-floorfree.py — floor-free PAIRED comparison of two candidate
captures against one shared reference capture.

WHY THIS EXISTS (2026-08-15, Qwen3.8-27B W8A8 publication decision)
-------------------------------------------------------------------
`kld-fullvocab.py compare` assigns every reference id that is ABSENT from a
candidate's stored top-K the candidate's tail mass spread uniformly over
(VOCAB - K). That floor is an ASSUMPTION, and its hit-rate is not constant
across arms:

    ROT-GPTQ 5.268%    v2-GPTQ 5.585%    v1 6.082%

The floor UNDER-estimates q_i for those ids and therefore OVER-estimates that
arm's KL, so the arm with more floor hits is penalised harder. The hit-rate
happens to track the ranking we want to publish (better arm -> fewer hits),
so the ranking cannot be defended from `compare` alone: part of the gap could
be the floor mechanism rather than model fidelity.

This tool removes the assumption instead of tuning it. It restricts the sum to
reference ids present in BOTH candidates' supports, so the two arms are scored
on IDENTICAL positions, with IDENTICAL p_i, over an IDENTICAL index set, and
every floored value is excluded from BOTH sides.

  partial(arm) = sum_{i in M} p_i * (log p_i - log q_i^arm)
  M = { i in topK(P) : i in supp(Q1) AND i in supp(Q2) }

⚠ READ THIS CORRECTLY: partial() is NOT a KL divergence. The support is
restricted and the retained p mass does not sum to 1, so the ABSOLUTE level is
meaningless and is always lower than the true KL. Only the DELTA between the
two arms, the paired t-statistic and the win-rate are interpretable. This is a
comparison statistic, not a fidelity measurement.

Because both arms are evaluated at the same positions, the paired test removes
position-level variance and is far more powerful than differencing two
independent means.

USAGE
-----
  python3 kld-paired-floorfree.py REF.npz ARM_A.npz ARM_B.npz --vocab 248320
"""
import argparse
import json
import math
import sys

import numpy as np


def load_matched(P_ids, P_lps, Qnpz, K, vocab):
    """Replicate kld-fullvocab.compare's alignment EXACTLY, but also return a
    boolean mask of which reference ids were genuinely FOUND in the candidate's
    support (i.e. did NOT fall back to the uniform tail floor)."""
    Qid = Qnpz["top_ids"].astype(np.int32)
    Qlp = Qnpz["top_lps"].astype(np.float32)

    so = np.argsort(Qid, axis=-1)
    qs_ids = np.take_along_axis(Qid, so, -1)
    qs_lps = np.take_along_axis(Qlp, so, -1)

    qt_each = Qnpz["tails"].astype(np.float32)[..., None] - math.log(vocab - K)
    n_inf_tail = int((~np.isfinite(qt_each)).sum())
    if n_inf_tail:
        qt_each = np.where(
            np.isfinite(qt_each), qt_each,
            math.log(float(np.finfo(np.float32).eps)) - math.log(vocab - K))

    qlp = np.broadcast_to(qt_each, P_lps.shape).copy()
    found = np.zeros(P_lps.shape, dtype=bool)

    for w in range(P_ids.shape[0]):
        ids = np.concatenate([qs_ids[w], P_ids[w]], axis=-1)
        order = np.argsort(ids, axis=-1, kind="stable")
        ids_s = np.take_along_axis(ids, order, -1)
        dup = ids_s[:, :-1] == ids_s[:, 1:]
        keep = dup & (order[:, :-1] < K) & (order[:, 1:] >= K)
        rows, spos = np.nonzero(keep)
        if rows.size:
            cols = order[rows, spos + 1] - K
            qlp[w, rows, cols] = qs_lps[w][rows, order[rows, spos]]
            found[w, rows, cols] = True

    valid = Qid[..., 0] >= 0
    return qlp, found, valid, n_inf_tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--vocab", type=int, default=248320)
    a = ap.parse_args()

    P = np.load(a.ref, allow_pickle=True)
    QA = np.load(a.arm_a, allow_pickle=True)
    QB = np.load(a.arm_b, allow_pickle=True)
    mp = json.loads(str(P["meta"]))
    ma = json.loads(str(QA["meta"]))
    mb = json.loads(str(QB["meta"]))

    assert (P["win_ids"] == QA["win_ids"]).all(), "arm A: different corpus"
    assert (P["win_ids"] == QB["win_ids"]).all(), "arm B: different corpus"

    Pid = P["top_ids"].astype(np.int32)
    Plp = P["top_lps"].astype(np.float32)
    K = int(Pid.shape[-1])
    p = np.exp(Plp)

    qlpA, foundA, validA, infA = load_matched(Pid, Plp, QA, K, a.vocab)
    qlpB, foundB, validB, infB = load_matched(Pid, Plp, QB, K, a.vocab)

    valid = (Pid[..., 0] >= 0) & validA & validB
    n = int(valid.sum())

    # the floor-free index set: present in BOTH candidates
    M = foundA & foundB

    covered = np.sum(p * M, axis=-1)[valid]           # retained p mass / position
    partA = np.sum(p * M * (Plp - qlpA), axis=-1)[valid]
    partB = np.sum(p * M * (Plp - qlpB), axis=-1)[valid]
    d = partB - partA                                  # >0 => arm A is better

    se = d.std(ddof=1) / math.sqrt(n)
    t = d.mean() / se if se > 0 else float("nan")
    winA = float((partA < partB).mean())

    # full-support numbers (WITH the floor) for side-by-side context
    fullA = np.maximum(np.sum(p * (Plp - qlpA), axis=-1), 0.0)[valid]
    fullB = np.maximum(np.sum(p * (Plp - qlpB), axis=-1), 0.0)[valid]

    tagA, tagB = ma["tag"], mb["tag"]
    print(f"ref={mp['tag']}  A={tagA}  B={tagB}  K={K}  positions={n}")
    if infA or infB:
        print(f"  ⚠ non-finite tails floored: A={infA} B={infB}")
    print()
    print(f"  index set M (present in BOTH arms): "
          f"{M[valid].sum() / (n * K) * 100:.3f}% of {n * K} gathered values")
    print(f"  retained reference mass           : "
          f"mean {covered.mean():.6f}  (min {covered.min():.4f})")
    print()
    print(f"  {'':22s} {'A=' + tagA:>22s} {'B=' + tagB:>22s}")
    print(f"  {'full support (floored)':22s} {fullA.mean():22.6f} {fullB.mean():22.6f}")
    print(f"  {'FLOOR-FREE partial':22s} {partA.mean():22.6f} {partB.mean():22.6f}")
    print()
    print(f"  paired delta (B - A)  : {d.mean():+.6f}  ± {se:.6f}   t = {t:+.1f}")
    print(f"  A better at           : {winA * 100:.2f}% of positions")
    print(f"  relative gap, floored : {(fullB.mean() - fullA.mean()) / fullA.mean() * 100:+.2f}%")
    print(f"  relative gap, FLOOR-FREE: {(partB.mean() - partA.mean()) / partA.mean() * 100:+.2f}%")
    print()
    if t > 3 and partA.mean() < partB.mean():
        print(f"  VERDICT: {tagA} leads with the floor mechanism REMOVED "
              f"(t={t:+.1f}) — the ranking is model fidelity, not a floor artifact.")
    elif t < -3:
        print(f"  VERDICT: RANKING REVERSES without the floor — {tagB} is better. "
              f"The floored comparison was an artifact. STOP and re-evaluate.")
    else:
        print(f"  VERDICT: INCONCLUSIVE without the floor (t={t:+.1f}). The arms are "
              f"not separable on shared support; do not publish the ranking as fact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
