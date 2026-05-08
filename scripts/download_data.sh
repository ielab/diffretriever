#!/bin/bash
# Download MS MARCO passage + TREC DL19/DL20 data.
#
# Usage:
#   bash scripts/download_data.sh            # all datasets
#   bash scripts/download_data.sh --msmarco  # MS MARCO only
#   bash scripts/download_data.sh --dl       # DL19 + DL20 only

MSMARCO=0
DL=0

if [[ $# -eq 0 ]]; then
    MSMARCO=1; DL=1
fi
while [[ $# -gt 0 ]]; do
    case $1 in
        --msmarco) MSMARCO=1; shift ;;
        --dl)      DL=1;      shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

get() {
    local url="$1" out="$2"
    echo "    downloading $(basename "$out") ..."
    wget -q --show-progress -O "$out" "$url" || { echo "    ERROR: failed to download $url" >&2; return 1; }
}

tsv_to_jsonl_queries() {
    python3 - "$1" "$2" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as fin, open(sys.argv[2], "w") as fout:
    for line in fin:
        qid, text = line.rstrip("\n").split("\t", 1)
        fout.write(json.dumps({"query_id": qid, "query": text}) + "\n")
PYEOF
}

# ── MS MARCO passage retrieval ──────────────────────────────────────────────
if [[ $MSMARCO -eq 1 ]]; then
    echo ">>> MS MARCO passage corpus..."
    mkdir -p data/msmarco

    if [[ ! -f data/msmarco/corpus.jsonl ]]; then
        get "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz" \
            data/msmarco/collection.tar.gz
        tar -xzf data/msmarco/collection.tar.gz -C data/msmarco/
        rm data/msmarco/collection.tar.gz
        python3 - <<'PYEOF'
import json
with open("data/msmarco/collection.tsv") as fin, \
     open("data/msmarco/corpus.jsonl", "w") as fout:
    for line in fin:
        pid, text = line.rstrip("\n").split("\t", 1)
        fout.write(json.dumps({"id": pid, "contents": text}) + "\n")
PYEOF
        echo "    corpus.jsonl ready"
    else
        echo "    corpus.jsonl already exists — skipping"
    fi

    echo ">>> MS MARCO dev queries..."
    if [[ ! -f data/msmarco/queries.dev.jsonl ]]; then
        get "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz" \
            /tmp/msmarco_queries.tar.gz
        tar -xzf /tmp/msmarco_queries.tar.gz -C data/msmarco/ queries.dev.tsv
        rm /tmp/msmarco_queries.tar.gz
        tsv_to_jsonl_queries data/msmarco/queries.dev.tsv data/msmarco/queries.dev.jsonl
        echo "    queries.dev.jsonl ready"
    else
        echo "    queries.dev.jsonl already exists — skipping"
    fi

    echo ">>> MS MARCO dev qrels..."
    if [[ ! -f data/msmarco/qrels.dev.tsv ]]; then
        get "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv" \
            data/msmarco/qrels.dev.tsv
        echo "    qrels.dev.tsv ready"
    else
        echo "    qrels.dev.tsv already exists — skipping"
    fi
fi

# ── TREC Deep Learning 2019 & 2020 ──────────────────────────────────────────
# Both use the MS MARCO passage corpus — only queries + qrels needed.
# Qrels from castorini/anserini-tools (more reliable than trec.nist.gov).
if [[ $DL -eq 1 ]]; then
    echo ">>> TREC DL19..."
    mkdir -p data/dl19

    if [[ ! -s data/dl19/queries.jsonl ]]; then
        get "https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.dl19-passage.txt" \
            data/dl19/queries.tsv && \
        tsv_to_jsonl_queries data/dl19/queries.tsv data/dl19/queries.jsonl && \
        echo "    dl19/queries.jsonl ready" || echo "    ERROR: failed to prepare dl19/queries.jsonl" >&2
    else
        echo "    dl19/queries.jsonl already exists — skipping"
    fi

    if [[ ! -f data/dl19/qrels.tsv ]]; then
        get "https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.dl19-passage.txt" \
            data/dl19/qrels.tsv
        echo "    dl19/qrels.tsv ready"
    else
        echo "    dl19/qrels.tsv already exists — skipping"
    fi

    echo ">>> TREC DL20..."
    mkdir -p data/dl20

    if [[ ! -s data/dl20/queries.jsonl ]]; then
        get "https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.dl20-passage.txt" \
            data/dl20/queries.tsv && \
        tsv_to_jsonl_queries data/dl20/queries.tsv data/dl20/queries.jsonl && \
        echo "    dl20/queries.jsonl ready" || echo "    ERROR: failed to prepare dl20/queries.jsonl" >&2
    else
        echo "    dl20/queries.jsonl already exists — skipping"
    fi

    if [[ ! -f data/dl20/qrels.tsv ]]; then
        get "https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.dl20-passage.txt" \
            data/dl20/qrels.tsv
        echo "    dl20/qrels.tsv ready"
    else
        echo "    dl20/qrels.tsv already exists — skipping"
    fi
fi

echo ">>> Done."
