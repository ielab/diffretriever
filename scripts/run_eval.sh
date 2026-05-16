#!/bin/bash
# =============================================================
# Minimal example: score encoded query/passage representations
# and compute MRR@10 / NDCG@10 with pytrec-eval.
#
# Assumes scripts/run_encode.sh has already produced query and
# corpus representations under ${RESULTS_DIR}/{queries,corpus}/.
#
# Edit the variables below, then:
#     bash scripts/run_eval.sh
#
# What it does:
#   evaluate_sweep.py runs all 5 retrieval modes in a single pass:
#     single_dense, multi_dense, sparse_max,
#     fusion_single_sparse_max, fusion_multi_sparse_max
#   and writes per-mode + summary metrics to ${OUTPUT_DIR}.
# =============================================================
set -euo pipefail

# --- Inputs -----------------------------------------------------
RESULTS_DIR=${RESULTS_DIR:-results/dream_few_K4/msmarco}
QRELS=${QRELS:-data/msmarco/qrels.dev.tsv}

# Encoded representations (defaults match run_encode.sh's output layout)
QUERY_DIR=${QUERY_DIR:-${RESULTS_DIR}/queries}
CORPUS_DIR=${CORPUS_DIR:-${RESULTS_DIR}/corpus}

# Where to write summary.json + {mode}.json + {mode}.trec
OUTPUT_DIR=${OUTPUT_DIR:-${RESULTS_DIR}/eval}

# --- Optional tuning -------------------------------------------
TOP_K=${TOP_K:-1000}                # retrieval depth
EVAL_TOP_K=${EVAL_TOP_K:-10}        # MRR / NDCG cutoff
REL_THRESHOLD=${REL_THRESHOLD:-1}   # qrel relevance threshold (use 2 for TREC DL)

# --- Validate ---------------------------------------------------
[[ -d "${QUERY_DIR}"  ]] || { echo "✗ ${QUERY_DIR} missing — run scripts/run_encode.sh first" >&2; exit 1; }
[[ -d "${CORPUS_DIR}" ]] || { echo "✗ ${CORPUS_DIR} missing — run scripts/run_encode.sh first" >&2; exit 1; }
[[ -f "${QRELS}"      ]] || { echo "✗ qrels missing: ${QRELS}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"

echo "═════════════════════════════════════════════════════════════"
echo "  ▶ Scoring all 5 retrieval modes"
echo "     query_dir   = ${QUERY_DIR}"
echo "     corpus_dir  = ${CORPUS_DIR}"
echo "     qrels       = ${QRELS}"
echo "     output_dir  = ${OUTPUT_DIR}"
echo "     top_k       = ${TOP_K}   eval_top_k = ${EVAL_TOP_K}   rel_threshold = ${REL_THRESHOLD}"
echo "═════════════════════════════════════════════════════════════"

python scripts/evaluate_sweep.py \
    --query_dir     "${QUERY_DIR}" \
    --corpus_dir    "${CORPUS_DIR}" \
    --qrels         "${QRELS}" \
    --output_dir    "${OUTPUT_DIR}" \
    --top_k         "${TOP_K}" \
    --eval_top_k    "${EVAL_TOP_K}" \
    --rel_threshold "${REL_THRESHOLD}"

echo
echo "✓ Done. ${OUTPUT_DIR} now contains:"
echo "    summary.json          (all 5 modes, average metrics)"
echo "    <mode>.json           (per-query MRR@${EVAL_TOP_K} / R@${TOP_K})"
echo "    <mode>.trec           (TREC run file)"
echo
echo "  Score a single TREC run with pytrec-eval directly:"
echo "    python scripts/eval_trec.py ${QRELS} ${OUTPUT_DIR}/multi_dense.trec \\"
echo "        --metrics mrr_cut_10 ndcg_cut_10"
