#!/usr/bin/env python3
"""Pre-tokenize Tevatron/msmarco-passage-aug for fast training.

Tokenizes every query + every positive/negative passage ONCE, offline.
Output: an HF datasets.Dataset on disk that can be loaded by
`MSMARCOTrainDataset` in --pretokenized mode (no per-step tokenization).

Each row in the output dataset has:
  - query_id          str
  - query_tokens      List[int]                          (truncated to --query_max_len)
  - query_content_ids List[int]                          (sparse content tokens)
  - positive_passages List[{docid, tokens, content_ids}] (text already title-prepended)
  - negative_passages List[{docid, tokens, content_ids}] (same)
  - length            int   total tokenized length (query + 16 passages, exact)

Run once per backbone tokenizer.  Different backbones (Llama, Qwen, Dream,
LLaDA) tokenize differently, so each gets its own pretokenized copy.

Usage
-----
    # For Llama / Qwen2.5 (AR backbones)
    python scripts/preprocess_msmarco_aug.py \\
        --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \\
        --output_dir data/msmarco-passage-aug-pretokenized/llama

    python scripts/preprocess_msmarco_aug.py \\
        --tokenizer Qwen/Qwen2.5-7B-Instruct \\
        --output_dir data/msmarco-passage-aug-pretokenized/qwen25

    # For Dream / LLaDA (diffusion backbones)
    python scripts/preprocess_msmarco_aug.py \\
        --tokenizer Dream-org/Dream-v0-Instruct-7B \\
        --output_dir data/msmarco-passage-aug-pretokenized/dream

    python scripts/preprocess_msmarco_aug.py \\
        --tokenizer GSAI-ML/LLaDA-8B-Instruct \\
        --output_dir data/msmarco-passage-aug-pretokenized/llada1

Then point `--train_data` at the corresponding pretokenized dir during training
(e.g. `--train_data data/msmarco-passage-aug-pretokenized/llama`).  The
`MSMARCOTrainDataset` auto-detects the pretokenized layout.

Wallclock: ~10-20 min one-time cost per backbone (~491k examples × ~16 passages
each = 8M sequences) on 16 CPU cores via num_proc.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def _load_stopwords() -> set:
    """Stopword + punctuation set — must match
    `src/models/sparse_utils.py:STOPWORDS` exactly:
        STOPWORDS = set(stopwords.words('english') + list(string.punctuation))
    Otherwise the precomputed `content_ids` would diverge from what the
    runtime tokenizer would have produced."""
    import string
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words('english') + list(string.punctuation))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            return set(stopwords.words('english') + list(string.punctuation))
    except Exception:
        return set(string.punctuation)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--tokenizer', required=True,
                    help='HF tokenizer id (e.g. meta-llama/Meta-Llama-3-8B-Instruct)')
    ap.add_argument('--input_dir', default=str(PROJECT_ROOT / 'data' / 'msmarco-passage-aug'),
                    help='Path to msmarco-passage-aug HF dataset on disk.')
    ap.add_argument('--output_dir', required=True,
                    help='Where to save the pretokenized dataset.')
    ap.add_argument('--query_max_len', type=int, default=32)
    ap.add_argument('--passage_max_len', type=int, default=156)
    ap.add_argument('--num_proc', type=int, default=8,
                    help='Parallel workers for map().  CPU-bound; 8-16 is plenty.')
    ap.add_argument('--max_examples', type=int, default=0,
                    help='Limit (for smoke testing).  0 = all.')
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[!] {output_dir} already exists and is non-empty.  "
              f"Delete it first to re-run.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print(f"Loading dataset from disk: {args.input_dir}", file=sys.stderr)
    from datasets import load_from_disk
    ds = load_from_disk(args.input_dir)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    print(f"  {len(ds)} examples", file=sys.stderr)

    stopwords = _load_stopwords()

    # Use NLTK's word_tokenize for content_ids extraction (matches the original
    # path in models/sparse_utils.py).  Fall back to .split() if NLTK unavailable.
    try:
        from nltk import word_tokenize
        try:
            word_tokenize('test')
        except LookupError:
            import nltk
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
    except ImportError:
        word_tokenize = None

    def _content_ids(text: str) -> List[int]:
        """Mirror `src/models/sparse_utils.py:get_content_token_ids` exactly.
        Returns a sorted list of unique token ids (set semantics, but list-
        encoded so HF datasets can store it).  Runtime converts back to set
        in the collator (`set(item['query']['content_ids'])`)."""
        if not text:
            return []
        words = (word_tokenize(text.lower()) if word_tokenize is not None
                 else text.lower().split())
        words = [w for w in words if w not in stopwords]
        if not words:
            return []
        # Batched encode — same call shape as get_content_token_ids
        batch_ids = tokenizer(words, add_special_tokens=False)['input_ids']
        ids: set = set()
        for sub in batch_ids:
            ids.update(sub)
        return sorted(ids)

    def _tokenize_batch(batch):
        out_query_tokens, out_query_content = [], []
        out_pos_passages, out_neg_passages = [], []
        out_lengths = []

        for q, qid, pos_list, neg_list in zip(
            batch['query'],
            batch.get('query_id', [str(i) for i in range(len(batch['query']))]),
            batch.get('positive_passages', [[]] * len(batch['query'])),
            batch.get('negative_passages', [[]] * len(batch['query'])),
        ):
            q = q or ''
            q_tokens = tokenizer.encode(
                q, add_special_tokens=False,
                truncation=True, max_length=args.query_max_len,
            )
            q_content = _content_ids(q)

            pos_out = []
            for p in (pos_list or []):
                title = p.get('title', '') or ''
                text = p.get('text', p.get('contents', '')) or ''
                full = f"{title} {text}".strip() if title else text
                p_tokens = tokenizer.encode(
                    full, add_special_tokens=False,
                    truncation=True, max_length=args.passage_max_len,
                )
                pos_out.append({
                    'docid':       p.get('docid', ''),
                    'tokens':      p_tokens,
                    'content_ids': _content_ids(full),
                })

            neg_out = []
            for p in (neg_list or []):
                title = p.get('title', '') or ''
                text = p.get('text', p.get('contents', '')) or ''
                full = f"{title} {text}".strip() if title else text
                p_tokens = tokenizer.encode(
                    full, add_special_tokens=False,
                    truncation=True, max_length=args.passage_max_len,
                )
                neg_out.append({
                    'docid':       p.get('docid', ''),
                    'tokens':      p_tokens,
                    'content_ids': _content_ids(full),
                })

            # Exact tokenized length (used for length-grouped sampling) —
            # query + all positives + all negatives.  This is the "true" sort
            # key, more accurate than the char-count proxy.
            tot_len = (len(q_tokens)
                       + sum(len(p['tokens']) for p in pos_out)
                       + sum(len(p['tokens']) for p in neg_out))

            out_query_tokens.append(q_tokens)
            out_query_content.append(q_content)
            out_pos_passages.append(pos_out)
            out_neg_passages.append(neg_out)
            out_lengths.append(tot_len)

        return {
            'query_id':          list(batch.get('query_id',
                                                [str(i) for i in range(len(batch['query']))])),
            'query_tokens':      out_query_tokens,
            'query_content_ids': out_query_content,
            'positive_passages': out_pos_passages,
            'negative_passages': out_neg_passages,
            'length':            out_lengths,
        }

    print(f"Tokenizing with num_proc={args.num_proc}...", file=sys.stderr)
    pretok = ds.map(
        _tokenize_batch,
        batched=True,
        batch_size=128,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        desc='pretokenize',
    )

    print(f"\nWriting to {output_dir}", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretok.save_to_disk(str(output_dir))
    print(f"Done.  Sample row keys: {list(pretok.features)}", file=sys.stderr)
    print(f"Length stats:  min={min(pretok['length'])}  "
          f"median={sorted(pretok['length'])[len(pretok)//2]}  "
          f"max={max(pretok['length'])}", file=sys.stderr)


if __name__ == '__main__':
    main()
