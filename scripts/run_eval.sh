#!/bin/bash
# =============================================================
# Minimal example: score encoded query/passage representations
# and compute MRR@10 / NDCG@10 with pytrec-eval.
#
# Assumes scripts/run_encode.sh has already produced query and
# corpus representations under ${RESULTS_DIR}.
#
# Edit the variables below, then:
#     bash scripts/run_eval.sh
# =============================================================
set -euo pipefail

# --- Inputs ----------------------------------------------------
RESULTS_DIR=${RESULTS_DIR:-results/dream_few_K4/msmarco}
QRELS=${QRELS:-data/msmarco/qrels.dev.tsv}

# --- Score modes to compute -----------------------------------
# all_modes: single_dense, multi_dense, sparse_max,
#            fusion_single_sparse_max, fusion_multi_sparse_max
SCORE_MODES=${SCORE_MODES:-multi_dense,sparse_max,fusion_multi_sparse_max}

# --- Run scoring + metrics -------------------------------------
python scripts/evaluate_sweep.py \
    --results_dir "${RESULTS_DIR}" \
    --qrels "${QRELS}" \
    --score_modes "${SCORE_MODES}"

# --- Or score a single run file with pytrec-eval directly ------
# python scripts/eval_trec.py \
#     --run "${RESULTS_DIR}/run.txt" \
#     --qrels "${QRELS}" \
#     --metrics mrr_cut_10 ndcg_cut_10
