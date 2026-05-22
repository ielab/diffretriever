#!/bin/bash
# =============================================================
# Minimal example: encode queries and a passage corpus using the
# representative-token interface (zero-shot DiffRetriever, zero-shot
# PromptReps, or a fine-tuned checkpoint).
#
# Edit the variables below, then:
#     bash scripts/run_encode.sh
#
# What it does:
#   1. Encodes queries  ←  ${QUERIES}  →  ${OUTPUT_DIR}/queries/
#   2. Encodes corpus   ←  ${CORPUS}   →  ${OUTPUT_DIR}/corpus/
#   Each side writes:  all_steps_shard_${SHARD_ID}.safetensors[.zst]
#   (the per-(query, K_q) and per-(passage, K_p) repr_hidden + sparse
#    activations consumed by scripts/run_eval.sh).
#
# For corpus-scale sharding (e.g. MS MARCO 8.8M passages on a cluster),
# fan out by submitting N copies with SHARD_ID=0..N-1 and NUM_SHARDS=N.
# =============================================================
set -euo pipefail

# --- Model -----------------------------------------------------
# Zero-shot diffusion: dream | llada1 | llada15 | llada2 | llada21
# Zero-shot AR:        llama | qwen | qwen25 | qwen3
# Fine-tuned:          trained_diff (pair with --model_dir)
#                      trained_ar   (pair with --model_dir)
# (Other model types supported by encode.py — diffembed_*, repllama —
#  are trained-only baselines; call encode.py directly for those.)
MODEL_TYPE=${MODEL_TYPE:-dream}
MODEL_NAME=${MODEL_NAME:-Dream-org/Dream-v0-Instruct-7B}
MODEL_DIR=${MODEL_DIR:-}                             # required for trained_* types

# --- Budget ----------------------------------------------------
# K = symmetric budget (same K for queries and corpus). To use the
# paper's asymmetric (K_q, K_p) setting, override K_Q and K_P directly:
#   K_Q=4 K_P=16 bash scripts/run_encode.sh
K=${K:-4}
K_Q=${K_Q:-$K}
K_P=${K_P:-$K}
NUM_DENOISE_STEPS=${NUM_DENOISE_STEPS:-1}            # diffusion only

# --- Prompts ---------------------------------------------------
PROMPT_VARIANT=${PROMPT_VARIANT:-few}                # few | three | one
QUERY_PROMPT=${QUERY_PROMPT:-prompts/default/query_prompt_${PROMPT_VARIANT}.yaml}
PASSAGE_PROMPT=${PASSAGE_PROMPT:-prompts/default/passage_prompt_${PROMPT_VARIANT}.yaml}

# --- Data ------------------------------------------------------
# Default file layouts produced by scripts/download_data.sh:
#   msmarco → data/msmarco/{queries.dev.jsonl, corpus.jsonl}
#   dl19/dl20 → data/${DATASET}/queries.jsonl  +  data/msmarco/corpus.jsonl (shared)
#   beir/<name> → data/beir/<name>/{queries.jsonl, corpus.jsonl}
DATASET=${DATASET:-msmarco}
case "${DATASET}" in
    msmarco)
        QUERIES=${QUERIES:-data/msmarco/queries.dev.jsonl}
        CORPUS=${CORPUS:-data/msmarco/corpus.jsonl}
        ;;
    dl19|dl20)
        QUERIES=${QUERIES:-data/${DATASET}/queries.jsonl}
        CORPUS=${CORPUS:-data/msmarco/corpus.jsonl}   # DL shares MSMARCO corpus
        ;;
    beir/*)
        QUERIES=${QUERIES:-data/${DATASET}/queries.jsonl}
        CORPUS=${CORPUS:-data/${DATASET}/corpus.jsonl}
        ;;
    *)
        QUERIES=${QUERIES:-data/${DATASET}/queries.jsonl}
        CORPUS=${CORPUS:-data/${DATASET}/corpus.jsonl}
        ;;
esac

OUTPUT_DIR=${OUTPUT_DIR:-results/${MODEL_TYPE}_${PROMPT_VARIANT}_K${K}/${DATASET}}

# --- Compute ---------------------------------------------------
BATCH_SIZE=${BATCH_SIZE:-64}                         # corpus batch
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-128}            # queries are short → larger batch
MAX_LENGTH=${MAX_LENGTH:-256}
SHARD_ID=${SHARD_ID:-0}                              # 0..NUM_SHARDS-1
NUM_SHARDS=${NUM_SHARDS:-1}

# --- Validate ---------------------------------------------------
[[ -f "${QUERIES}" ]] || { echo "✗ queries file missing: ${QUERIES}" >&2; exit 1; }
[[ -f "${CORPUS}"  ]] || { echo "✗ corpus file missing:  ${CORPUS}"  >&2; exit 1; }
[[ -f "${QUERY_PROMPT}"   ]] || { echo "✗ query prompt missing:   ${QUERY_PROMPT}"   >&2; exit 1; }
[[ -f "${PASSAGE_PROMPT}" ]] || { echo "✗ passage prompt missing: ${PASSAGE_PROMPT}" >&2; exit 1; }

# --- Identify model (HF id vs local checkpoint dir) -------------
MODEL_REF=()
if [[ "${MODEL_TYPE}" == "trained_diff" || "${MODEL_TYPE}" == "trained_ar" ]]; then
    [[ -z "${MODEL_DIR}" ]] && { echo "✗ Set MODEL_DIR for ${MODEL_TYPE}" >&2; exit 1; }
    MODEL_REF=(--model_dir "${MODEL_DIR}")
else
    [[ -z "${MODEL_NAME}" ]] && { echo "✗ Set MODEL_NAME for ${MODEL_TYPE}" >&2; exit 1; }
    MODEL_REF=(--model_name_or_path "${MODEL_NAME}")
fi

# --- Plumb K through the right flag depending on backbone -------
# Diffusion (+ trained_diff) → --n_gen_tokens K --num_denoise_steps S
# Trained AR                 → --n_gen_tokens K   (no denoise steps)
# Zero-shot AR (llama/qwen)  → --num_pooled_tokens K  (K=1 → 0, single-representation)
ar_pooled() { [[ "$1" == "1" ]] && echo "0" || echo "$1"; }
case "${MODEL_TYPE}" in
    dream|llada1|llada15|llada2|llada21|trained_diff)
        Q_KFLAG=(--n_gen_tokens "${K_Q}" --num_denoise_steps "${NUM_DENOISE_STEPS}")
        P_KFLAG=(--n_gen_tokens "${K_P}" --num_denoise_steps "${NUM_DENOISE_STEPS}")
        ;;
    trained_ar)
        Q_KFLAG=(--n_gen_tokens "${K_Q}")
        P_KFLAG=(--n_gen_tokens "${K_P}")
        ;;
    llama|qwen|qwen25|qwen3)
        Q_KFLAG=(--num_pooled_tokens "$(ar_pooled "${K_Q}")")
        P_KFLAG=(--num_pooled_tokens "$(ar_pooled "${K_P}")")
        ;;
    *)
        echo "✗ Unsupported MODEL_TYPE=${MODEL_TYPE}" >&2; exit 1 ;;
esac

# --- Common flags ----------------------------------------------
COMMON=(
    --model_type "${MODEL_TYPE}"
    "${MODEL_REF[@]}"
    --query_prompt   "${QUERY_PROMPT}"
    --passage_prompt "${PASSAGE_PROMPT}"
    --encode_type all_steps
    --sparse_topk 256
    --max_length "${MAX_LENGTH}"
    --shard_id   "${SHARD_ID}"
    --num_shards "${NUM_SHARDS}"
)

# --- Encode queries (passes --is_query so the query prompt is applied) ---
echo
echo "═════════════════════════════════════════════════════════════"
echo "  ▶ Queries   ${QUERIES}"
echo "             →  ${OUTPUT_DIR}/queries   (K_q=${K_Q}, batch=${QUERY_BATCH_SIZE})"
echo "═════════════════════════════════════════════════════════════"
python scripts/encode.py \
    "${COMMON[@]}" \
    --input_file "${QUERIES}" \
    --output_dir "${OUTPUT_DIR}/queries" \
    --is_query \
    --batch_size "${QUERY_BATCH_SIZE}" \
    "${Q_KFLAG[@]}"

# --- Encode corpus (no --is_query → passage prompt is applied) ----------
echo
echo "═════════════════════════════════════════════════════════════"
echo "  ▶ Corpus    ${CORPUS}"
echo "             →  ${OUTPUT_DIR}/corpus    (K_p=${K_P}, batch=${BATCH_SIZE})"
echo "═════════════════════════════════════════════════════════════"
python scripts/encode.py \
    "${COMMON[@]}" \
    --input_file "${CORPUS}" \
    --output_dir "${OUTPUT_DIR}/corpus" \
    --batch_size "${BATCH_SIZE}" \
    "${P_KFLAG[@]}"

echo
echo "✓ Encoded both sides under ${OUTPUT_DIR}/"
echo "  Next:  RESULTS_DIR=${OUTPUT_DIR} bash scripts/run_eval.sh"
