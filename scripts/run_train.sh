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
# =============================================================
set -euo pipefail

# --- Backbone -------------------------------------------------
MODEL_TYPE=${MODEL_TYPE:-dream}                    # dream | llada1 | llada15 | llada2
MODEL_NAME=${MODEL_NAME:-Dream-org/Dream-v0-Instruct-7B}

# --- Budget (K_q, K_p) ---------------------------------------
K_Q=${K_Q:-4}
K_P=${K_P:-16}
NUM_DENOISE_STEPS=${NUM_DENOISE_STEPS:-1}

# --- Data + output -------------------------------------------
TRAIN_DATA=${TRAIN_DATA:-data/msmarco-passage-aug}      # Tevatron augmented triples
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/diffretriever-${MODEL_TYPE}-kq${K_Q}-kp${K_P}}

# --- Training -------------------------------------------------
NUM_GPUS=${NUM_GPUS:-4}
BATCH_PER_GPU=${BATCH_PER_GPU:-32}                       # global batch = NUM_GPUS * BATCH_PER_GPU
GRAD_ACCUM=${GRAD_ACCUM:-1}
EPOCHS=${EPOCHS:-1}
LR=${LR:-1e-4}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-64}
TEMPERATURE=${TEMPERATURE:-0.01}
N_HARD_NEGS=${N_HARD_NEGS:-15}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-configs/ds_zero2.json}

# --- Launch ---------------------------------------------------
deepspeed --num_gpus="${NUM_GPUS}" scripts/train_retriever.py \
    --model_type "${MODEL_TYPE}" \
    --model_name_or_path "${MODEL_NAME}" \
    --train_data "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_gen_q_tokens "${K_Q}" \
    --n_gen_p_tokens "${K_P}" \
    --num_denoise_steps "${NUM_DENOISE_STEPS}" \
    --per_device_train_batch_size "${BATCH_PER_GPU}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --temperature "${TEMPERATURE}" \
    --train_n_passages $((1 + N_HARD_NEGS)) \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --bf16 \
    --logging_steps 50 \
    --save_strategy epoch \
    --report_to wandb
