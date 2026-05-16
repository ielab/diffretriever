#!/bin/bash
# =============================================================
# Minimal example: contrastive fine-tuning of DiffRetriever on
# MS MARCO augmented triples (Tevatron format).
#
# Edit the variables below for your setup, then:
#     bash scripts/run_train.sh
#
# For the autoregressive PromptReps and the re-trained baselines
# (DiffEmbed, RepLLaMA), use the same pattern with their training
# scripts: train_ar_retriever.py, train_diffembed.py,
# train_repllama.py.
#
# NOTE: train_retriever.py constructs HF TrainingArguments internally
#       (bf16=True, report_to='wandb', save_strategy='steps'), so those
#       flags are NOT passed on the CLI — only the argparse-defined
#       hyperparameters below.
# =============================================================
set -euo pipefail

# --- Backbone -------------------------------------------------
MODEL_TYPE=${MODEL_TYPE:-dream}                    # dream | llada1 | llada15 | llada2
MODEL_NAME=${MODEL_NAME:-Dream-org/Dream-v0-Instruct-7B}

# --- Budget (K_q, K_p) ---------------------------------------
K_Q=${K_Q:-4}
K_P=${K_P:-16}
NUM_DENOISE_STEPS=${NUM_DENOISE_STEPS:-1}

# --- Prompts (required by train_retriever.py) ----------------
PROMPT_VARIANT=${PROMPT_VARIANT:-few}              # few | one | three
QUERY_PROMPT=${QUERY_PROMPT:-prompts/default/query_prompt_${PROMPT_VARIANT}.yaml}
PASSAGE_PROMPT=${PASSAGE_PROMPT:-prompts/default/passage_prompt_${PROMPT_VARIANT}.yaml}

# --- Data + output -------------------------------------------
# Default points at the HF dataset slug (Tevatron's augmented triples are
# fetched + cached on first use). Set to a local pre-tokenized dir if you
# pre-processed with scripts/preprocess_msmarco_aug.py.
TRAIN_DATA=${TRAIN_DATA:-Tevatron/msmarco-passage-aug}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/diffretriever-${MODEL_TYPE}-kq${K_Q}-kp${K_P}}

# --- Training -------------------------------------------------
NUM_GPUS=${NUM_GPUS:-4}
BATCH_PER_GPU=${BATCH_PER_GPU:-32}                  # global batch = NUM_GPUS * BATCH_PER_GPU * GRAD_ACCUM
GRAD_ACCUM=${GRAD_ACCUM:-1}
EPOCHS=${EPOCHS:-1}
LR=${LR:-1e-4}                                      # paper: 1e-4 for LoRA, 1e-5 for full FT
LORA_RANK=${LORA_RANK:-16}                          # 0 = full fine-tune
LORA_ALPHA=${LORA_ALPHA:-64}
TEMPERATURE=${TEMPERATURE:-0.01}
N_HARD_NEGS=${N_HARD_NEGS:-15}                      # train_retriever.py: --n_negatives (positive is added implicitly)
SPARSE_WEIGHT=${SPARSE_WEIGHT:-1.0}
DENOISING_WEIGHT=${DENOISING_WEIGHT:-0.0}           # >0 → diffusion-native denoising auxiliary
DIVERSITY_WEIGHT=${DIVERSITY_WEIGHT:-0.0}           # >0 → multi-vector orthogonality hinge
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-configs/ds_zero2.json}

# --- Validate -------------------------------------------------
[[ -f "${QUERY_PROMPT}"   ]] || { echo "✗ ${QUERY_PROMPT} missing"   >&2; exit 1; }
[[ -f "${PASSAGE_PROMPT}" ]] || { echo "✗ ${PASSAGE_PROMPT} missing" >&2; exit 1; }
[[ -f "${DEEPSPEED_CONFIG}" ]] || { echo "✗ ${DEEPSPEED_CONFIG} missing" >&2; exit 1; }

# --- Launch ---------------------------------------------------
deepspeed --num_gpus="${NUM_GPUS}" scripts/train_retriever.py \
    --model_type "${MODEL_TYPE}" \
    --model_name "${MODEL_NAME}" \
    --query_prompt   "${QUERY_PROMPT}" \
    --passage_prompt "${PASSAGE_PROMPT}" \
    --train_data "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_gen_q_tokens "${K_Q}" \
    --n_gen_p_tokens "${K_P}" \
    --num_denoise_steps "${NUM_DENOISE_STEPS}" \
    --per_device_batch_size "${BATCH_PER_GPU}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --temperature "${TEMPERATURE}" \
    --n_negatives "${N_HARD_NEGS}" \
    --sparse_weight "${SPARSE_WEIGHT}" \
    --denoising_weight "${DENOISING_WEIGHT}" \
    --diversity_weight "${DIVERSITY_WEIGHT}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --logging_steps 50
