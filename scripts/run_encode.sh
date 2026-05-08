#!/bin/bash
# =============================================================
# Minimal example: encode queries and a passage corpus using the
# representative-token interface (zero-shot DiffRetriever, zero-shot
# PromptReps, or a fine-tuned checkpoint).
#
# Edit the variables below, then:
#     bash scripts/run_encode.sh
#
# Output layout (per shard):
#   ${OUTPUT_DIR}/queries/{repr_hidden,repr_logits}.{safetensors,jsonl}
#   ${OUTPUT_DIR}/corpus/{repr_hidden,repr_logits}.{safetensors,jsonl}
# =============================================================
set -euo pipefail

# --- Model -----------------------------------------------------
# Zero-shot diffusion: dream | llada1 | llada15 | llada2
# Zero-shot AR:        llama | qwen25 | qwen3
# Fine-tuned:          trained_diff (pair with --model_dir)
#                      trained_ar   (pair with --model_dir)
MODEL_TYPE=${MODEL_TYPE:-dream}
MODEL_NAME=${MODEL_NAME:-Dream-org/Dream-v0-Instruct-7B}
MODEL_DIR=${MODEL_DIR:-}                            # for trained_* types

# --- Budget ----------------------------------------------------
K=${K:-4}                                           # number of representative tokens
NUM_DENOISE_STEPS=${NUM_DENOISE_STEPS:-1}

# --- Prompt variant -------------------------------------------
PROMPT_VARIANT=${PROMPT_VARIANT:-few}                # few | three | one
PASSAGE_PROMPT="prompts/default/passage_prompt_${PROMPT_VARIANT}.yaml"
QUERY_PROMPT="prompts/default/query_prompt_${PROMPT_VARIANT}.yaml"

# --- Data ------------------------------------------------------
DATASET=${DATASET:-msmarco}                          # msmarco | dl19 | dl20 | beir/<name>
QUERIES=${QUERIES:-data/${DATASET}/queries.tsv}
CORPUS=${CORPUS:-data/${DATASET}/corpus.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-results/${MODEL_TYPE}_${PROMPT_VARIANT}_K${K}/${DATASET}}

# --- Compute ---------------------------------------------------
BATCH_SIZE=${BATCH_SIZE:-64}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-128}
MAX_LENGTH=${MAX_LENGTH:-256}

# --- Launch ----------------------------------------------------
EXTRA_ARGS=""
if [[ "${MODEL_TYPE}" == "trained_diff" || "${MODEL_TYPE}" == "trained_ar" ]]; then
    [[ -z "${MODEL_DIR}" ]] && { echo "Set MODEL_DIR for ${MODEL_TYPE}" >&2; exit 1; }
    EXTRA_ARGS="--model_dir ${MODEL_DIR}"
fi

python scripts/encode_promptreps.py \
    --model_type "${MODEL_TYPE}" \
    ${MODEL_NAME:+--model_name_or_path "${MODEL_NAME}"} \
    ${EXTRA_ARGS} \
    --queries "${QUERIES}" \
    --corpus "${CORPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    --query_prompt "${QUERY_PROMPT}" \
    --passage_prompt "${PASSAGE_PROMPT}" \
    --n_gen_tokens "${K}" \
    --num_denoise_steps "${NUM_DENOISE_STEPS}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --query_batch_size "${QUERY_BATCH_SIZE}"
