#!/usr/bin/env python3
"""
Download and prepare MS MARCO passage dev data from HuggingFace.

Saves:
  data/msmarco/corpus.jsonl      — 8.8M passages
  data/msmarco/queries.dev.jsonl — ~6980 dev queries
  data/msmarco/qrels.dev.tsv    — dev qrels (TREC format)

Usage:
    python scripts/prepare_msmarco.py
    python scripts/prepare_msmarco.py --max_passages 100000  # small subset
"""

import argparse
import json
import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/msmarco')
    parser.add_argument('--max_passages', type=int, default=0,
                        help='Max passages to save (0=all). Use 100000 for quick testing.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from datasets import load_dataset

    # --- Corpus ---
    corpus_path = os.path.join(args.output_dir, 'corpus.jsonl')
    if os.path.exists(corpus_path):
        logger.info(f"Corpus already exists at {corpus_path}, skipping.")
    else:
        logger.info("Loading MS MARCO passage corpus from HuggingFace...")
        corpus = load_dataset('Tevatron/msmarco-passage-corpus', split='train')

        logger.info(f"Saving corpus to {corpus_path}...")
        count = 0
        with open(corpus_path, 'w') as f:
            for item in corpus:
                doc = {
                    'id': item['docid'],
                    'text': f"{item.get('title', '')} {item['text']}".strip(),
                }
                f.write(json.dumps(doc) + '\n')
                count += 1
                if args.max_passages > 0 and count >= args.max_passages:
                    break
                if count % 500000 == 0:
                    logger.info(f"  {count} passages...")
        logger.info(f"Saved {count} passages to {corpus_path}")

    # --- Dev queries ---
    queries_path = os.path.join(args.output_dir, 'queries.dev.jsonl')
    if os.path.exists(queries_path):
        logger.info(f"Queries already exist at {queries_path}, skipping.")
    else:
        logger.info("Loading MS MARCO dev queries from HuggingFace...")
        dev = load_dataset('Tevatron/msmarco-passage', split='validation')

        query_count = 0
        seen_queries = set()
        with open(queries_path, 'w') as fq:
            for item in dev:
                qid = item['query_id']
                if qid not in seen_queries:
                    seen_queries.add(qid)
                    fq.write(json.dumps({'id': qid, 'text': item['query']}) + '\n')
                    query_count += 1
        logger.info(f"Saved {query_count} queries to {queries_path}")

    # --- Dev qrels (download official MS MARCO qrels) ---
    qrels_path = os.path.join(args.output_dir, 'qrels.dev.tsv')
    if os.path.exists(qrels_path) and os.path.getsize(qrels_path) > 0:
        logger.info(f"Qrels already exist at {qrels_path}, skipping.")
    else:
        import urllib.request
        qrels_url = 'https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv'
        logger.info(f"Downloading official MS MARCO dev qrels from {qrels_url}...")
        urllib.request.urlretrieve(qrels_url, qrels_path)
        qrel_count = sum(1 for _ in open(qrels_path))
        logger.info(f"Saved {qrel_count} qrels to {qrels_path}")

    logger.info("Done!")


if __name__ == '__main__':
    main()
