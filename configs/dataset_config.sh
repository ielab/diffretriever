#!/bin/bash
# ── Single source of truth for per-dataset encoding config ───────────────────
# Sourced by: encode.sh, submit_sweep.sh, encode_local.sh, eval_local.sh, monitor.sh
#
# Each entry: "shards wall_minutes"
# Sized so one shard finishes well within the wall limit.
# Multi-step (s>=2) doubles shards (halving docs/shard) so wall time stays the same.
#
# Shard counts reflect the **post-combine** state (combine_shards.py).
# corpus_ready() trusts MERGE_INFO.json when present, so these numbers
# only matter for not-yet-combined corpora; setting them to the typical
# post-combine count keeps `--corpus` re-encoding from triggering on a
# combined corpus that lacks MERGE_INFO for some reason.
#
#   Dataset              Corpus size   Shards (post-combine)
#   ─────────────────    ───────────   ─────────────────────
#   msmarco              8.8M          35
#   msmarco (--small)    ~88K          10
#   beir/hotpotqa        5.2M          15
#   beir/fever           5.4M          15
#   beir/nq              2.7M          20
#   beir/dbpedia-entity  4.6M          20
#   beir/climate-fever   5.4M          20
#   beir/quora           523K          10
#   beir/trec-covid      172K          7
#   beir/fiqa            57K           2
#   beir/cqadupstack     ~40K          2
#   beir/webis-touche20  382K          5
#   beir/scifact         5K            1
#   beir/arguana         8K            1
#   beir/*  (others)     <10K          1

dataset_encode_config() {
    local dataset="$1"
    # Each shard count = (original_count) × small_int.  Keeping shard counts as
    # integer multiples of the original lets `combine_shards.py` merge clean
    # `(orig_count)` partitions post-encoding.  Per-shard wall ≤ ~7 min on H100
    # at ~200 docs/s; walltime cap is set just above expected runtime so SLURM
    # fairshare priority stays high (shorter walltime → scheduled sooner).
    #
    # original_count → multiplier → resulting shards
    case "$dataset" in
        #                          shards  wall_mins
        msmarco)              echo "105      15" ;;   # 35×3 → 84k/shard ≈ 7 min
        beir/hotpotqa)        echo "75       12" ;;   # 15×5 → 69k/shard ≈ 6 min
        beir/fever)           echo "75       13" ;;   # 15×5 → 72k/shard ≈ 6 min
        beir/nq)              echo "60       11" ;;    # 20×3 → 45k/shard ≈ 4 min
        beir/dbpedia-entity)  echo "80       15" ;;   # 20×4 → 57k/shard ≈ 5 min
        beir/climate-fever)   echo "80       15" ;;   # 20×4 → 67k/shard ≈ 6 min
        beir/quora)           echo "10       9" ;;    # 10×1 → 52k/shard ≈ 4 min
        beir/trec-covid)      echo "7        12" ;;    # 7×1  → 25k/shard ≈ 2 min
        beir/fiqa)            echo "2        11" ;;    # 2×1  → 28k/shard ≈ 2 min
        beir/cqadupstack)     echo "2        6" ;;    # 2×1  → 20k/shard ≈ 2 min
        beir/scifact)         echo "1        9" ;;    # 1×1  →  5K total ≈ 30s
        beir/arguana)         echo "1        9" ;;    # 1×1  →  8K total ≈ 45s
        beir/webis-touche2020) echo "5       8" ;;    # 5×1  → 76k/shard ≈ 6 min
        beir/*)               echo "1        8" ;;
        *)                    echo "1        8" ;;
    esac
}

# Convenience: get just the shard count for a dataset + steps + K combo.
# Usage:
#   num_shards=$(dataset_num_shards "beir/hotpotqa" 2)        # K assumed 4
#   num_shards=$(dataset_num_shards "beir/hotpotqa" 1 0 16)   # K_p=16 corpus
#
# 4th arg is the corpus-side K (n_gen_p_tokens for asymmetric, n_gen_tokens
# otherwise).  K>=16 doubles per-shard cost ~linearly with K, just like s>=2
# does ~linearly with steps — both push past the wall budget calibrated for
# K=4 s=1.  Doubling kicks in when EITHER condition holds (we don't 4× when
# both — the wall has built-in slack and 4× hurts SLURM throughput).
dataset_num_shards() {
    local dataset="$1" steps="${2:-1}" small="${3:-0}" K="${4:-4}"
    if [[ $small -eq 1 ]]; then
        echo 10
        return
    fi
    local cfg shards
    cfg=$(dataset_encode_config "$dataset")
    shards=$(echo "$cfg" | awk '{print $1}')
    # Double shards on the larger corpora when per-shard cost is high.
    # Limited to msmarco + the larger BEIRs we evaluate on (nq, hotpotqa,
    # trec-covid).  Small BEIRs (fiqa/scifact/arguana/quora) keep their
    # original count — they finish within wall even at s=2 / K=16.
    local high_cost=0
    [[ $steps -ge 2 ]] && high_cost=1
    [[ $K -ge 16 ]]    && high_cost=1
    if [[ $high_cost -eq 1 ]]; then
        case "$dataset" in
            msmarco|beir/nq|beir/hotpotqa|beir/trec-covid)
                shards=$((shards * 2)) ;;
        esac
    fi
    echo "$shards"
}

# Convenience: get wall time in minutes for a dataset + steps combo.
# Shards are already doubled for multi-step (see dataset_num_shards), which
# halves docs/shard.  The extra forward passes per doc roughly cancel out,
# so wall time per shard stays ~constant regardless of steps.
# Usage: wall_mins=$(dataset_wall_mins "msmarco" 2)  → 10
dataset_wall_mins() {
    local dataset="$1" steps="${2:-1}"
    local cfg mins
    cfg=$(dataset_encode_config "$dataset")
    mins=$(echo "$cfg" | awk '{print $2}')
    echo "$mins"
}
