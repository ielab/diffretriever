#!/usr/bin/env python3
"""Evaluate a TREC run file against qrels using pytrec_eval.

Defaults: MRR@10, nDCG@10, R@1000 — the standard MSMARCO / DL / BEIR set.
Denominator is `len(qrels)` (qids with at least one judgement), matching
MSMARCO official MRR semantics: queries the system retrieved nothing for
contribute 0, not get dropped.

Usage
-----
  # MSMARCO dev (binary, MRR@10 is the headline)
  python scripts/eval_trec.py \\
      data/msmarco/qrels.dev.tsv \\
      results/.../single_dense.trec

  # TREC-DL19/20 (graded; standard convention is rel >= 2 → relevant)
  python scripts/eval_trec.py \\
      data/dl19/qrels.tsv results/.../fusion_multi_sparse_max.trec \\
      --rel-threshold 2

  # BEIR (any qrels.tsv with 3- or 4-col format)
  python scripts/eval_trec.py \\
      data/beir/scifact/qrels/test.tsv results/.../single_dense.trec

  # Custom metric set
  python scripts/eval_trec.py qrels.tsv run.trec \\
      --metrics ndcg_cut.10 ndcg_cut.100 recall.1000 map

Both qrels formats are accepted:
  TREC 4-col:  qid 0 did rel
  BEIR 3-col:  qid did rel    (with optional header)
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path


def load_qrels(path: Path) -> dict:
    """Return {qid: {did: rel}} keeping ALL judgements (positive and negative)
    so pytrec_eval can compute graded nDCG / MAP correctly.  rel_threshold
    is applied later via RelevanceEvaluator's `relevance_level` argument."""
    qrels: dict[str, dict[str, int]] = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:                                   # TREC 4-col
                qid, did, rel = parts[0], parts[2], parts[3]
            elif len(parts) == 3:                                 # BEIR 3-col
                qid, did, rel = parts[0], parts[1], parts[2]
            else:
                continue
            try:
                qrels[qid][did] = int(rel)
            except ValueError:
                continue                                          # skip header
    return dict(qrels)


def load_run(path: Path) -> dict:
    """Return {qid: {did: score}} from a TREC 6-col run file (no truncation)."""
    run: dict[str, dict[str, float]] = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, did, _rank, score_s, _ = parts[:6]
            try:
                run[qid][did] = float(score_s)
            except ValueError:
                continue
    return dict(run)


def truncate_run(run: dict, k: int) -> dict:
    """Return a copy of `run` keeping only the top-k highest-scoring docs
    per query.  Used for MRR@K (recip_rank has no built-in cutoff)."""
    out = {}
    for qid, scores in run.items():
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        out[qid] = dict(top)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('qrels', type=Path, help='Qrels file (3- or 4-column)')
    ap.add_argument('run', type=Path, help='TREC run file (6-column)')
    ap.add_argument('--rel-threshold', type=int, default=1,
                    help='Min rel value to count as relevant (default 1; '
                         'use 2 for TREC-DL19/20).')
    ap.add_argument('--mrr-cutoff', type=int, default=10,
                    help='Cutoff for MRR (default 10 — MSMARCO standard).')
    ap.add_argument('--metrics', nargs='+', default=None,
                    help='Extra pytrec_eval metric names beyond MRR@cutoff. '
                         'Default: ndcg_cut.10 recall.1000.')
    ap.add_argument('--per-query', action='store_true',
                    help='Also print per-query metrics.')
    args = ap.parse_args()

    try:
        import pytrec_eval
    except ImportError:
        sys.exit("pytrec_eval not installed.  pip install pytrec_eval "
                 "(or pytrec-eval-terrier).")

    if args.metrics is None:
        args.metrics = ['ndcg_cut.10', 'recall.1000']

    qrels = load_qrels(args.qrels)
    if not qrels:
        sys.exit(f"No qrels loaded from {args.qrels}")
    run_full = load_run(args.run)
    if not run_full:
        sys.exit(f"No hits loaded from {args.run}")

    # MRR@K needs a separate evaluator on a truncated run because
    # `recip_rank` has no built-in cutoff in pytrec_eval.
    run_topk = truncate_run(run_full, args.mrr_cutoff)
    e_mrr = pytrec_eval.RelevanceEvaluator(
        qrels, {'recip_rank'}, relevance_level=args.rel_threshold,
    ).evaluate(run_topk)

    # All other metrics use the full run; their cutoffs are baked into
    # the metric name (e.g. ndcg_cut.10, recall.1000).
    e_other = pytrec_eval.RelevanceEvaluator(
        qrels, set(args.metrics), relevance_level=args.rel_threshold,
    ).evaluate(run_full)

    n_qrels = len(qrels)
    n_run = len(run_full)
    n_scored = len(set(e_mrr) | set(e_other))

    print(f"qrels:   {args.qrels}  ({n_qrels} qids)")
    print(f"run:     {args.run}  ({n_run} qids)")
    print(f"rel ≥ {args.rel_threshold}, scored {n_scored}/{n_qrels} qids")
    print('-' * 60)

    # MRR@K — the key pytrec_eval returns is always 'recip_rank'.
    mrr = sum(s.get('recip_rank', 0.0) for s in e_mrr.values()) / n_qrels
    print(f"  MRR@{args.mrr_cutoff:<11}  {mrr:.4f}")

    # Other metrics — iterate over whatever keys pytrec_eval actually
    # returned (handles dot/underscore normalization across versions).
    if e_other:
        all_keys: set[str] = set()
        for s in e_other.values():
            all_keys.update(s.keys())
        for k in sorted(all_keys):
            total = sum(s.get(k, 0.0) for s in e_other.values())
            print(f"  {k:<14}  {total / n_qrels:.4f}")

    if args.per_query:
        print('-' * 60)
        print('per-query:')
        for qid in sorted(set(e_mrr) | set(e_other)):
            mrr_q = e_mrr.get(qid, {}).get('recip_rank', 0.0)
            other = e_other.get(qid, {})
            row = f'mrr@{args.mrr_cutoff}={mrr_q:.4f}  ' + '  '.join(
                f'{k}={v:.4f}' for k, v in sorted(other.items()))
            print(f'  {qid}: {row}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
