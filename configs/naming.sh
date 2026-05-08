#!/bin/bash
# Shared helper: detect trained-checkpoint metadata + compose output config name.
# Sourced by: scripts/encode.sh, scripts/encode_local.sh
#
# Design (simplified):
#   The output config name is just `trained_{diff|ar}_<basename(model_dir)>`.
#   train.sh is responsible for naming the model dir with everything we need
#   to disambiguate runs (K_q/K_p, steps, lora, sparse weight, etc.). The
#   encoder/eval pipeline just trusts that name verbatim — no regex parsing,
#   no risk of duplicating `_k4_s2` when the dir already says `-kq4-kp16-s2`.
#
#   K / steps / backbone are still extracted from the saved config JSON (and
#   from the path) because downstream encoder CLI flags need them, but they
#   no longer participate in TRAINED_CONFIG.

# trained_detect_config <model_dir>
#
# Sets globals:
#   TRAINED_TYPE       = trained_diff | trained_ar
#   TRAINED_BACKBONE   = dream | llada1 | llada15 | llada2 | llama | qwen25 | qwen3 | qwen
#   TRAINED_K          = n_gen_tokens (diff) or n_pooled_tokens (AR)  — informational / CLI hint
#   TRAINED_K_Q        = corpus-side K (n_gen_q_tokens for diff, else TRAINED_K)
#   TRAINED_K_P        = passage/corpus-side K (n_gen_p_tokens for diff, else TRAINED_K)
#   TRAINED_STEPS      = num_denoise_steps (diff) or 1 (AR)
#   TRAINED_CONFIG     = trained_{diff|ar}_<basename>[-ckptN]
#
# Returns 1 if the dir has no final training config → caller should skip.
trained_detect_config() {
    local md="$1"

    # ── Which kind of trained model is this? ─────────────────────────────────
    if [[ -f "$md/retriever_config.json" ]]; then
        TRAINED_TYPE="trained_diff"
        TRAINED_K=$(python3 -c "import json; print(json.load(open('$md/retriever_config.json'))['n_gen_tokens'])")
        TRAINED_STEPS=$(python3 -c "import json; print(json.load(open('$md/retriever_config.json'))['num_denoise_steps'])")
        # Asymmetric: corpus uses n_gen_p_tokens (defaults to n_gen_tokens
        # for symmetric models that don't store the per-side fields).
        TRAINED_K_P=$(python3 -c "import json; d=json.load(open('$md/retriever_config.json')); print(d.get('n_gen_p_tokens') or d['n_gen_tokens'])")
        TRAINED_K_Q=$(python3 -c "import json; d=json.load(open('$md/retriever_config.json')); print(d.get('n_gen_q_tokens') or d['n_gen_tokens'])")
    elif [[ -f "$md/ar_retriever_config.json" ]]; then
        TRAINED_TYPE="trained_ar"
        TRAINED_K=$(python3 -c "import json; print(json.load(open('$md/ar_retriever_config.json'))['n_pooled_tokens'])")
        TRAINED_STEPS=1
        TRAINED_K_P="$TRAINED_K"
        TRAINED_K_Q="$TRAINED_K"
    elif [[ -f "$md/diffembed_config.json" ]]; then
        # DiffEmbed (mean-pool, K=1) — single-vector by construction.
        TRAINED_TYPE="trained_diffembed"
        TRAINED_BACKBONE=$(python3 -c "import json; print(json.load(open('$md/diffembed_config.json'))['model_type'])")
        TRAINED_K=1
        TRAINED_STEPS=1
        TRAINED_K_P=1
        TRAINED_K_Q=1
    elif [[ -f "$md/repllama_config.json" ]]; then
        # RepLLaMA (last-token EOS pool, K=1) — AR analog of DiffEmbed.
        TRAINED_TYPE="trained_repllama"
        TRAINED_BACKBONE="llama"
        TRAINED_K=1
        TRAINED_STEPS=1
        TRAINED_K_P=1
        TRAINED_K_Q=1
    elif [[ "$(basename "$md")" == repllama_* || "$(basename "$md")" == repllama-* ]] \
         && compgen -G "$md/checkpoint-*" > /dev/null; then
        # In-progress RepLLaMA training (final config not written yet, but
        # checkpoint-N/ subdirs exist).  K and STEPS are fixed for RepLLaMA,
        # so we can compose a config name from the directory pattern alone.
        TRAINED_TYPE="trained_repllama"
        TRAINED_BACKBONE="llama"
        TRAINED_K=1
        TRAINED_STEPS=1
        TRAINED_K_P=1
        TRAINED_K_Q=1
    else
        return 1
    fi

    # ── Backbone: infer from the path. Order matters — longer names first
    # so "qwen25" matches before "qwen", and "llada15" before "llada1".
    if [[ "$TRAINED_TYPE" == "trained_diff" ]]; then
        if   [[ "$md" == *llada2*  ]]; then TRAINED_BACKBONE="llada2"
        elif [[ "$md" == *llada15* ]]; then TRAINED_BACKBONE="llada15"
        elif [[ "$md" == *llada1*  ]]; then TRAINED_BACKBONE="llada1"
        else                                TRAINED_BACKBONE="dream"
        fi
    elif [[ "$TRAINED_TYPE" == "trained_diffembed" ]]; then
        : # already set from diffembed_config.json above
    elif [[ "$TRAINED_TYPE" == "trained_repllama" ]]; then
        : # backbone is llama by construction (set from repllama_config.json)
    else
        if   [[ "$md" == *qwen25* || "$md" == *Qwen2.5* ]]; then TRAINED_BACKBONE="qwen25"
        elif [[ "$md" == *qwen3*  || "$md" == *Qwen3*   ]]; then TRAINED_BACKBONE="qwen3"
        elif [[ "$md" == *qwen*   || "$md" == *Qwen*    ]]; then TRAINED_BACKBONE="qwen"
        else                                                      TRAINED_BACKBONE="llama"
        fi
    fi

    # ── Handle intermediate HF Trainer subdirs (.../checkpoint-N/)
    local mb=$(basename "$md") ckpt=""
    if [[ "$mb" =~ ^checkpoint-([0-9]+)$ ]]; then
        ckpt="-ckpt${BASH_REMATCH[1]}"
        mb=$(basename "$(dirname "$md")")
    fi

    # ── Compose: just trained_{diff|ar|diffembed|repllama}_<basename>[-ckptN]
    # train.sh / train_diffembed.sh / train_repllama.sh already encode everything
    # (K, lora, etc.) into the basename via NAME_STEM, so we trust it verbatim.
    # For DiffEmbed / RepLLaMA: strip the redundant prefix from the basename
    # (else we'd get "trained_diffembed_diffembed_dream-lora16").
    local stripped_mb="$mb"
    if [[ "$TRAINED_TYPE" == "trained_diffembed" && "$mb" == diffembed_* ]]; then
        stripped_mb="${mb#diffembed_}"
    elif [[ "$TRAINED_TYPE" == "trained_repllama" && "$mb" == repllama_* ]]; then
        stripped_mb="${mb#repllama_}"
    fi
    TRAINED_CONFIG="${TRAINED_TYPE}_${stripped_mb}${ckpt}"
    return 0
}
