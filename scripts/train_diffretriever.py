#!/usr/bin/env python3
"""
Train a diffusion retriever (Dream / LLaDA1 / LLaDA1.5 / LLaDA2) on MS MARCO.

  - InfoNCE loss; multi-vector MaxSim over K generation tokens (K tested: 1..16)
  - In-batch negatives + hard negatives (Tevatron-style augmented data)
  - Optional FLOPS-L1 sparse regularization (`--sparse_weight`)
  - Optional multi-step denoising during training (`--num_denoise_steps`)
  - LoRA or full-fine-tuning; PromptReps-style prompt template

Encoding: PromptReps (K appended MASK tokens, single bidirectional forward pass,
fully differentiable). Supports end-to-end training of both the dense (mean-pool)
and sparse (logit-max-pool over vocabulary) heads.

Usage (4-GPU LoRA, matches the paper recipe):
    torchrun --nproc_per_node 4 scripts/train_diffretriever.py \\
        --model_name Dream-org/Dream-v0-Instruct-7B \\
        --model_type dream \\
        --query_prompt  prompts/default/query_prompt_few.yaml \\
        --passage_prompt prompts/default/passage_prompt_few.yaml \\
        --train_data     Tevatron/msmarco-passage-aug \\
        --output_dir     models_new/dream_few-k4-s1-lora16-sp1.0 \\
        --n_gen_tokens   4  --num_denoise_steps 1 \\
        --n_negatives    15 \\
        --per_device_batch_size 8  --gradient_accumulation_steps 4 \\
        --learning_rate  1e-4 \\
        --num_train_epochs 3 \\
        --lora_rank 16 --lora_alpha 64

    # Local data instead of HuggingFace:
    ... --train_data data/msmarco/train.jsonl
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# CPU-only tokenizer (picklable → runs in DataLoader worker processes)
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalTokenizer:
    """
    Holds just the CPU tokenization state extracted from a
    TrainableDiffusionRetriever. Fully picklable so DataLoader workers
    can tokenize in parallel instead of blocking the main process.
    """

    def __init__(self, model: 'TrainableDiffusionRetriever'):
        self.tokenizer = model.tokenizer
        self.query_prefix_ids = list(model._query_prefix_ids)
        self.query_suffix_ids = list(model._query_suffix_ids)
        self.passage_prefix_ids = list(model._passage_prefix_ids)
        self.passage_suffix_ids = list(model._passage_suffix_ids)
        self.mask_token_id = model.mask_token_id
        self.n_gen_tokens = model.n_gen_tokens
        # Per-side K for asymmetric training (e.g. K_q=4, K_p=16).
        # Falls back to n_gen_tokens for symmetric configs.
        self.n_gen_q_tokens = getattr(model, 'n_gen_q_tokens', model.n_gen_tokens)
        self.n_gen_p_tokens = getattr(model, 'n_gen_p_tokens', model.n_gen_tokens)
        # Structural tail [", chat_end, EOS] — model.forward() reads gen reps at
        # positions L - K - n_tail, so the sequence MUST end with these tail
        # tokens for training positions to line up with zero-shot inference.
        self.tail_ids = list(model._tail_ids)
        self.query_max_length = getattr(model, 'query_max_length', model.max_length)
        self.passage_max_length = getattr(model, 'passage_max_length', model.max_length)
        self.pad_token_id = (model.tokenizer.pad_token_id
                             or model.tokenizer.eos_token_id or 0)
        self.sparse_weight = model.sparse_weight

    def __call__(self, text: str, is_query: bool) -> Dict:
        prefix_ids = self.query_prefix_ids if is_query else self.passage_prefix_ids
        suffix_ids = self.query_suffix_ids if is_query else self.passage_suffix_ids
        k_side = self.n_gen_q_tokens if is_query else self.n_gen_p_tokens
        mask_block = [self.mask_token_id] * k_side + self.tail_ids
        max_len = self.query_max_length if is_query else self.passage_max_length
        # PromptReps semantic: `max_len` bounds the TEXT ONLY (prefix, suffix,
        # MASK tokens, and structural tail are all appended on top). This matches
        # PromptReps/dataset.py which truncates the raw text to `query_max_len`
        # before prepending/appending prompt templates. So --query_max_len 32
        # means "32 tokens of actual query text", not "32 tokens total".
        # (Zero-shot encoders use the subtract-from-total semantic with max_len=512,
        # where the distinction is irrelevant because real texts are shorter.)
        max_text_len = max_len
        enc = self.tokenizer(
            [text],
            padding=False,
            truncation=True,
            max_length=max_text_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        ids = prefix_ids + enc['input_ids'][0] + suffix_ids + mask_block
        result = {'input_ids': ids, 'attention_mask': [1] * len(ids)}
        if self.sparse_weight > 0:
            if not hasattr(self, '_get_content_ids'):
                from models.sparse_utils import get_content_token_ids
                self._get_content_ids = get_content_token_ids
            result['content_ids'] = list(self._get_content_ids([text], self.tokenizer)[0])
        return result

    def from_pretokenized(self, text_tokens, content_ids, is_query: bool) -> Dict:
        """Same layout as __call__ but skips the tokenizer.encode call when
        text_tokens has already been pre-computed offline."""
        prefix_ids = self.query_prefix_ids if is_query else self.passage_prefix_ids
        suffix_ids = self.query_suffix_ids if is_query else self.passage_suffix_ids
        k_side = self.n_gen_q_tokens if is_query else self.n_gen_p_tokens
        mask_block = [self.mask_token_id] * k_side + self.tail_ids
        ids = list(prefix_ids) + list(text_tokens) + list(suffix_ids) + mask_block
        result = {'input_ids': ids, 'attention_mask': [1] * len(ids)}
        if self.sparse_weight > 0:
            result['content_ids'] = list(content_ids)
        return result

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from models.diffretriever_trainable import TrainableDiffusionRetriever

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MSMARCOTrainDataset(Dataset):
    """
    Loads MS MARCO training triples.

    Accepts either:
    - HuggingFace dataset name (e.g. "Tevatron/msmarco-passage")
    - Local JSONL file (each line: {"query": ..., "positive_passages": [...],
                                    "negative_passages": [...]})
    """

    def __init__(
        self,
        data_source: str,
        n_negatives: int = 7,
        max_examples: int = 0,
        seed: int = 42,
        split: str = 'train',
        offset: int = 0,
        ret_tokenizer: Optional[RetrievalTokenizer] = None,
        epoch: int = 0,
    ):
        self.n_negatives = n_negatives
        self.seed = seed                       # used for per-qid hashed seed
        self.epoch = epoch                     # rotates negatives across epochs
        self.ret_tokenizer = ret_tokenizer

        p = Path(data_source)
        if p.is_file():
            logger.info(f"Loading data from JSONL: {data_source}")
            self.data = []
            with open(data_source) as f:
                for i, line in enumerate(f):
                    if i < offset:
                        continue
                    self.data.append(json.loads(line))
                    if max_examples > 0 and len(self.data) >= max_examples:
                        break
        elif p.is_dir():
            logger.info(f"Loading data from disk: {data_source}")
            from datasets import load_from_disk
            ds = load_from_disk(data_source)
            end = offset + max_examples if max_examples > 0 else len(ds)
            self.data = ds.select(range(offset, min(end, len(ds))))
        else:
            logger.info(f"Loading data from HuggingFace: {data_source} "
                        f"(split={split}, offset={offset})")
            from datasets import load_dataset
            ds = load_dataset(data_source, split=split)
            end = offset + max_examples if max_examples > 0 else len(ds)
            self.data = ds.select(range(offset, min(end, len(ds))))

        # Detect pre-tokenized layout produced by
        # scripts/preprocess_msmarco_aug.py (`query_tokens` column instead of
        # raw `query` text).  Skips per-call tokenization in __getitem__.
        cols = (self.data.column_names if hasattr(self.data, 'column_names')
                else (list(self.data[0].keys()) if self.data else []))
        self.pretokenized = 'query_tokens' in cols
        if self.pretokenized:
            logger.info("Detected pre-tokenized dataset → skipping per-call "
                        "tokenization (faster DataLoader, exact lengths).")

        logger.info(f"Loaded {len(self.data)} examples (split={split}, offset={offset})")

        # Per-example length for length-grouped sampling.  Pretokenized:
        # exact tokens.  Otherwise: char-count proxy.
        self.lengths = self._compute_lengths()
        logger.info(f"Length stats: min={min(self.lengths)} "
                    f"median={sorted(self.lengths)[len(self.lengths)//2]} "
                    f"max={max(self.lengths)}"
                    f"{'  (exact tokens)' if self.pretokenized else '  (char-count proxy)'}")

    def _compute_lengths(self) -> List[int]:
        """Length per example.  Pretokenized: read 'length' column.  Otherwise
        fall back to char-count proxy (rough but correlated)."""
        if self.pretokenized and hasattr(self.data, 'column_names') \
                and 'length' in self.data.column_names:
            return list(self.data['length'])
        if self.pretokenized:
            return [int(item.get('length', 0)) for item in self.data]

        out = []
        if hasattr(self.data, 'column_names'):
            queries = self.data['query']
            try:
                positives = self.data['positive_passages']
            except (KeyError, ValueError):
                positives = [[] for _ in queries]
            try:
                negatives = self.data['negative_passages']
            except (KeyError, ValueError):
                negatives = [[] for _ in queries]
            for q, pos, neg in zip(queries, positives, negatives):
                qlen = len(q or '')
                plen = sum(len(p.get('text', '') or '') for p in (pos or []))
                nlen = sum(len(p.get('text', '') or '') for p in (neg or []))
                out.append(qlen + plen + nlen)
        else:
            for item in self.data:
                qlen = len(item.get('query', '') or '')
                plen = sum(len(p.get('text', '') or '')
                           for p in item.get('positive_passages', []) or [])
                nlen = sum(len(p.get('text', '') or '')
                           for p in item.get('negative_passages', []) or [])
                out.append(qlen + plen + nlen)
        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.data[idx]

        # Per-qid stable RNG (PromptReps `_hashed_seed` semantics).
        qid = item.get('query_id') or item.get('qid') or str(idx)
        item_seed = (self.seed * 1_000_003 + hash(str(qid))) & 0xFFFFFFFF
        rng = random.Random(item_seed)

        # ── Pretokenized fast path ────────────────────────────────────────
        if self.pretokenized:
            q_tokens = list(item['query_tokens'])
            q_content_ids = list(item.get('query_content_ids', []))

            pos_list = list(item.get('positive_passages') or [])
            if not pos_list:
                p_pos = {'tokens': [], 'content_ids': []}
            else:
                p_pos = pos_list[(item_seed + self.epoch) % len(pos_list)]

            neg_list = list(item.get('negative_passages') or [])
            n_neg = self.n_negatives
            sampled_negs = []
            if neg_list:
                shuffled = list(neg_list)
                rng.shuffle(shuffled)
                if len(shuffled) >= n_neg:
                    offset_n = (self.epoch * n_neg) % len(shuffled)
                    sampled_negs = (shuffled + shuffled)[offset_n: offset_n + n_neg]
                else:
                    sampled_negs = [shuffled[rng.randrange(len(shuffled))]
                                    for _ in range(n_neg)]
            while len(sampled_negs) < n_neg:
                sampled_negs.append(sampled_negs[-1] if sampled_negs
                                     else {'tokens': [], 'content_ids': []})

            chosen = [p_pos] + sampled_negs
            if self.ret_tokenizer is not None:
                return {
                    'query':    self.ret_tokenizer.from_pretokenized(
                        q_tokens, q_content_ids, is_query=True),
                    'passages': [self.ret_tokenizer.from_pretokenized(
                        p['tokens'], p.get('content_ids', []), is_query=False)
                        for p in chosen],
                }
            raise RuntimeError("ret_tokenizer is required for pretokenized data")

        # ── Raw-text path (legacy) ─────────────────────────────────────────
        query = item['query']
        pos_list = item.get('positive_passages', [])
        neg_list = list(item.get('negative_passages') or [])

        if not pos_list:
            pos_text = item.get('positive', '')
        else:
            # Rotate through positives across epochs (PromptReps-style).
            pos = pos_list[(item_seed + self.epoch) % len(pos_list)]
            title = pos.get('title', '')
            text = pos.get('text', pos.get('contents', ''))
            pos_text = f"{title} {text}".strip() if title else text

        # Negatives: shuffle deterministically per-qid, take an epoch-offset
        # window of n_negatives.  Wraps around for short pools.
        n_neg = self.n_negatives
        sampled_negs = []
        if neg_list:
            shuffled = list(neg_list)
            rng.shuffle(shuffled)
            if len(shuffled) >= n_neg:
                offset_n = (self.epoch * n_neg) % len(shuffled)
                doubled = shuffled + shuffled
                sampled_negs = doubled[offset_n: offset_n + n_neg]
            else:
                sampled_negs = [shuffled[rng.randrange(len(shuffled))]
                                for _ in range(n_neg)]
        neg_texts = []
        for neg in sampled_negs:
            title = neg.get('title', '')
            text = neg.get('text', neg.get('contents', ''))
            neg_texts.append(f"{title} {text}".strip() if title else text)

        while len(neg_texts) < self.n_negatives:
            neg_texts.append(neg_texts[-1] if neg_texts else '')

        if self.ret_tokenizer is not None:
            # Tokenize in the worker process (parallel, non-blocking for GPU)
            return {
                'query': self.ret_tokenizer(query, is_query=True),
                'passages': [self.ret_tokenizer(p, is_query=False)
                             for p in [pos_text] + neg_texts],
            }
        return {
            'query': query,
            'passages': [pos_text] + neg_texts,  # 1 positive + n_negatives
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collator
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalCollator:
    """
    Pads pre-tokenized queries/passages produced by RetrievalTokenizer in workers.
    Falls back to on-the-fly tokenization via the model if items are raw strings
    (e.g. when ret_tokenizer was not passed to the dataset).
    """

    def __init__(self, model: TrainableDiffusionRetriever,
                 pad_token_id: Optional[int] = None):
        self.model = model
        self.pad_token_id = (pad_token_id
                             or model.tokenizer.pad_token_id
                             or model.tokenizer.eos_token_id or 0)

    def _pad(self, encs: List[Dict], pad_to_multiple_of: int = 8) -> tuple:
        # Round up to multiple of 8 → H100 tensor cores stay aligned in bf16.
        # The extra pad tokens are masked out, so the math is unchanged.
        raw_max = max(len(e['input_ids']) for e in encs)
        if pad_to_multiple_of > 1:
            max_len = ((raw_max + pad_to_multiple_of - 1)
                       // pad_to_multiple_of * pad_to_multiple_of)
        else:
            max_len = raw_max
        ids = torch.full((len(encs), max_len), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(encs), max_len, dtype=torch.long)
        for i, e in enumerate(encs):
            l = len(e['input_ids'])
            # Left-pad: content right-aligned so MASK tokens are always at the end
            ids[i, max_len - l:] = torch.tensor(e['input_ids'], dtype=torch.long)
            mask[i, max_len - l:] = torch.tensor(e['attention_mask'], dtype=torch.long)
        return ids, mask

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Fast path: items already tokenized by workers
        if isinstance(batch[0]['query'], dict):
            q_encs = [item['query'] for item in batch]
            p_encs = [p for item in batch for p in item['passages']]
            q_ids, q_mask = self._pad(q_encs)
            p_ids, p_mask = self._pad(p_encs)
        else:
            # Fallback: raw strings, tokenize here
            queries = [item['query'] for item in batch]
            passages = [p for item in batch for p in item['passages']]
            q_ids, q_mask = self.model.tokenize(queries, is_query=True)
            p_ids, p_mask = self.model.tokenize(passages, is_query=False)

        result = {
            'query_input_ids': q_ids,
            'query_attention_mask': q_mask,
            'passage_input_ids': p_ids,
            'passage_attention_mask': p_mask,
        }
        # Pass content token IDs for sparse filtering (pre-computed in workers)
        if isinstance(batch[0]['query'], dict) and 'content_ids' in batch[0]['query']:
            result['query_content_ids'] = [set(item['query']['content_ids']) for item in batch]
            result['passage_content_ids'] = [set(p['content_ids'])
                                             for item in batch for p in item['passages']]
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalTrainer(Trainer):
    """HF Trainer wrapper that handles the dict-output model."""

    def _get_train_sampler(self, train_dataset=None):
        """When `group_by_length=True`, Trainer's stock sampler only reads
        from a `datasets.Dataset` column — but we wrap it in a custom torch
        Dataset.  Feed `train_dataset.lengths` directly to LengthGroupedSampler.
        """
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if ds is None:
            return None
        if self.args.group_by_length and hasattr(ds, 'lengths'):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size
                           * self.args.gradient_accumulation_steps,
                dataset=ds,
                lengths=list(ds.lengths),
                model_input_name=None,
            )
        return super()._get_train_sampler(train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_dict = model(**inputs)
        loss = loss_dict['loss']
        return (loss, loss_dict) if return_outputs else loss

    def log(self, logs: Dict, *args, **kwargs):
        # Surface per-step losses from the last batch if available
        if hasattr(self, '_last_loss_dict'):
            for k, v in self._last_loss_dict.items():
                if k != 'loss' and isinstance(v, torch.Tensor):
                    logs[k] = v.item()
        super().log(logs, *args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, loss_dict = self.compute_loss(model, inputs, return_outputs=True)

        self._last_loss_dict = loss_dict

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if not torch.isfinite(loss):
            print(f">>> WARNING: non-finite loss ({loss.item()}), skipping step", flush=True)
            return loss.new_zeros(1).squeeze()

        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train Diffusion Retriever')

    # Model
    parser.add_argument('--model_name', required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--model_type', required=True, choices=['dream', 'llada1', 'llada15', 'llada2'])
    parser.add_argument('--query_prompt', required=True,
                        help='Path to query YAML prompt')
    parser.add_argument('--passage_prompt', required=True,
                        help='Path to passage YAML prompt')

    # Encoding
    parser.add_argument('--n_gen_tokens', type=int, default=4,
                        help='K: number of MASK generation tokens (symmetric default; '
                             'overridden per-side by --n_gen_q_tokens / --n_gen_p_tokens)')
    parser.add_argument('--n_gen_q_tokens', type=int, default=None,
                        help='K_q: query-side MASK count (default: same as --n_gen_tokens). '
                             'Use to train asymmetric retrieval (e.g. K_q=4, K_p=16).')
    parser.add_argument('--n_gen_p_tokens', type=int, default=None,
                        help='K_p: passage-side MASK count (default: same as --n_gen_tokens).')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Legacy single max length (overridden by query/passage_max_length)')
    parser.add_argument('--query_max_length', type=int, default=None)
    parser.add_argument('--passage_max_length', type=int, default=None)
    parser.add_argument('--normalize', action='store_true', default=True)

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=0,
                        help='LoRA rank (0 = full fine-tuning)')
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Data
    parser.add_argument('--train_data', default='Tevatron/msmarco-passage-aug',
                        help='HuggingFace dataset name or path to local JSONL')
    parser.add_argument('--n_negatives', type=int, default=7,
                        help='Number of hard negatives per query')
    parser.add_argument('--max_train_examples', type=int, default=0,
                        help='Max training examples (0 = all)')
    parser.add_argument('--data_seed', type=int, default=42)

    # Eval
    parser.add_argument('--eval_data', default=None,
                        help='Eval data source (default: same as train_data)')
    parser.add_argument('--eval_split', default='train',
                        help='HF dataset split for eval (default: train)')
    parser.add_argument('--eval_offset', type=int, default=500000,
                        help='Skip first N examples for eval (avoid train overlap)')
    parser.add_argument('--max_eval_examples', type=int, default=500,
                        help='Max eval examples')
    parser.add_argument('--eval_steps', type=int, default=None,
                        help='Eval every N optimizer steps (None = no eval)')

    # Loss
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--dense_weight', type=float, default=1.0,
                        help='Weight for dense retrieval loss (0 = sparse-only training)')
    parser.add_argument('--num_denoise_steps', type=int, default=None,
                        help='Denoising steps for step-aligned encoding '
                             '(default: same as n_gen_tokens)')
    parser.add_argument('--sparse_weight', type=float, default=1.0,
                        help='Weight for sparse InfoNCE loss (0 = disabled)')
    parser.add_argument('--denoising_weight', type=float, default=0.0,
                        help='Weight for diffusion-native denoising auxiliary loss (0 = disabled)')
    parser.add_argument('--diversity_weight', type=float, default=0.0,
                        help='Weight for multi-vector diversity loss (0 = disabled)')
    parser.add_argument('--progressive_step_weight', type=float, default=0.0,
                        help='Weight for progressive step supervision in multi-step training '
                             '(0 = disabled, only final step gets loss). '
                             'Applies retrieval loss at each denoising step with linearly '
                             'increasing weight (t/T). Only effective when num_denoise_steps > 1.')
    parser.add_argument('--denoise_mask_ratio', type=float, default=0.15,
                        help='Fraction of text tokens to mask for denoising loss')
    parser.add_argument('--disable_hidden_hook', action='store_true',
                        help='Force output_hidden_states=True instead of adapter-specific hidden hook. '
                             'Useful for debugging LLaDA hidden extraction.')
    parser.add_argument('--debug_dense_metrics', action='store_true',
                        help='Log dense score-gap and MASK-vector diagnostics in training logs.')
    parser.add_argument('--debug_compare_hidden_once', action='store_true',
                        help='On the first forward pass, compare the hidden hook output against the '
                             'true last hidden state and log the absolute difference.')
    parser.add_argument('--fresh_final', action='store_true',
                        help='At the final denoising step, use fresh hidden states for ALL K '
                             'positions (not frozen states from earlier steps). Only effective '
                             'when num_denoise_steps > 1.')
    parser.add_argument('--soft_denoising', action='store_true',
                        help='Use soft-token multi-step denoising (differentiable embeddings '
                             'instead of hard argmax replacement). Enables full gradient flow '
                             'through all K positions at every step. Only effective when '
                             'num_denoise_steps > 1.')
    parser.add_argument('--soft_temperature', type=float, default=1.0,
                        help='Temperature for softmax in soft-token denoising (lower = sharper)')
    parser.add_argument('--corruption_rate', type=float, default=0.0,
                        help='Max text corruption rate for denoising-conditioned training. '
                             'Randomly masks 0..rate fraction of passage text tokens per batch. '
                             '0 = disabled. Recommended: 0.15-0.20.')

    # Training
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--per_device_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_scheduler_type', default='cosine')
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--dataloader_num_workers', type=int, default=8,
                        help='Tokenisation + get_content_token_ids run in workers. '
                             'At 100%% GPU util the data pipeline is already not the '
                             'bottleneck; 8 workers/rank (32 total across 4 GPUs) is '
                             'plenty and leaves CPU / RAM headroom for other jobs.')
    parser.add_argument('--dataloader_prefetch_factor', type=int, default=4,
                        help='How many batches each worker pre-queues (default HF=2).')
    parser.add_argument('--dataloader_persistent_workers', action='store_true', default=True,
                        help='Keep workers alive between epochs (saves re-spawn cost).')
    parser.add_argument('--deepspeed', default=None,
                        help='Path to DeepSpeed config JSON')
    parser.add_argument('--no_gradient_checkpointing', action='store_true',
                        help='Disable gradient checkpointing (uses more memory)')
    parser.add_argument('--group_by_length', action='store_true', default=False,
                        help='Group similar-length sequences into the same batch '
                             'to reduce padding waste. ~5-15%% speedup, no recipe '
                             'change.')
    # ── K-adapter (joint K-router + retriever training, train-time-loss teacher)
    parser.add_argument('--use_k_adapter', action='store_true', default=False,
                        help='Train a tiny K-routing head jointly with the retriever. '
                             'Encode at K_max, compute per-cell InfoNCE losses, use '
                             'softmax(-loss/τ_T) as the adapter teacher (no MRR labels).')
    parser.add_argument('--adapter_weight', type=float, default=1.0,
                        help='λ on the KL(teacher || adapter) loss (default 1.0).')
    parser.add_argument('--teacher_temperature', type=float, default=1.0,
                        help='τ_T for the per-query teacher distribution.  Lower = '
                             'sharper teacher (commits to lowest-loss cell faster).')
    parser.add_argument('--k_adapter_options', type=str, default=None,
                        help='Comma-separated K options for the adapter '
                             '(default: factors of K_max in {1,2,4,8,16}).')
    # ── K pre-encoder (two-stage variable-length training) ───────────
    parser.add_argument('--use_k_pre_encoder', action='store_true', default=False,
                        help='Predict K_q and K_p per item via a small MLP head '
                             'over the embedding-layer output BEFORE the main '
                             'encoder runs.  Each item is sliced to its predicted '
                             'K, batches padded to max-K-in-batch — main encoder '
                             'runs at variable length, giving real encoding savings.')
    parser.add_argument('--gumbel_temperature', type=float, default=1.0,
                        help='τ for Gumbel-softmax sampling.  Lower = harder '
                             'commitments to sampled K.')
    parser.add_argument('--k_cost_lambda', type=float, default=0.001,
                        help='λ on E[K_q + K_p] regularizer.  Discourages '
                             'collapsing to K_max.  Set 0 to disable.')
    parser.add_argument('--k_pre_encoder_options', type=str, default=None,
                        help='Comma-separated K options for the pre-encoder '
                             '(default: {1,2,4,8,16} ≤ K_max).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_from_checkpoint', default=None)
    parser.add_argument('--resume_weights_only', action='store_true',
                        help='Load model weights from checkpoint but skip DeepSpeed '
                             'optimizer states. Use when changing GPU count after '
                             'a ZeRO world-size mismatch error.')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_denoise_steps is None:
        # Use the larger side: that's the side with more positions to decode.
        k_q = args.n_gen_q_tokens if args.n_gen_q_tokens is not None else args.n_gen_tokens
        k_p = args.n_gen_p_tokens if args.n_gen_p_tokens is not None else args.n_gen_tokens
        args.num_denoise_steps = max(k_q, k_p)

    # H100: enable TF32 for matmul/cuDNN — ~2x speedup vs FP32 with negligible
    # accuracy loss (bf16 activations already dominant; TF32 affects accumulation)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Suppress forked-tokenizer parallelism warnings from DataLoader workers
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    logger.info(f"Building model: {args.model_name} ({args.model_type})")
    q_max = args.query_max_length or args.max_length
    p_max = args.passage_max_length or args.max_length

    # Parse k_adapter_options string (e.g. "1,2,4,8,16") if user passed it
    k_opts = None
    if args.k_adapter_options:
        k_opts = tuple(int(x) for x in args.k_adapter_options.split(','))
    # Parse k_pre_encoder_options string
    k_pe_opts = None
    if args.k_pre_encoder_options:
        k_pe_opts = tuple(int(x) for x in args.k_pre_encoder_options.split(','))

    model = TrainableDiffusionRetriever.from_backbone(
        model_name=args.model_name,
        model_type=args.model_type,
        query_prompt=args.query_prompt,
        passage_prompt=args.passage_prompt,
        max_length=p_max,
        n_gen_tokens=args.n_gen_tokens,
        n_gen_q_tokens=args.n_gen_q_tokens,
        n_gen_p_tokens=args.n_gen_p_tokens,
        temperature=args.temperature,
        num_denoise_steps=args.num_denoise_steps,
        sparse_weight=args.sparse_weight,
        normalize=args.normalize,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        disable_hidden_hook=args.disable_hidden_hook,
        # K-adapter
        use_k_adapter=args.use_k_adapter,
        adapter_weight=args.adapter_weight,
        teacher_temperature=args.teacher_temperature,
        k_adapter_options=k_opts,
        # K pre-encoder
        use_k_pre_encoder=args.use_k_pre_encoder,
        gumbel_temperature=args.gumbel_temperature,
        k_cost_lambda=args.k_cost_lambda,
        k_pre_encoder_options=k_pe_opts,
    )
    model.query_max_length = q_max
    model.passage_max_length = p_max
    model.denoising_weight = args.denoising_weight
    model.diversity_weight = args.diversity_weight
    model.denoise_mask_ratio = args.denoise_mask_ratio
    model.progressive_step_weight = args.progressive_step_weight
    model.dense_weight = args.dense_weight
    model.debug_dense_metrics = args.debug_dense_metrics
    model.debug_compare_hidden_once = args.debug_compare_hidden_once
    model.use_fresh_final = args.fresh_final
    model.soft_denoising = args.soft_denoising
    model.soft_temperature = args.soft_temperature
    model.corruption_rate = args.corruption_rate

    if args.dense_weight != 1.0:
        logger.info(f"Dense loss weight: {args.dense_weight}")
    if args.denoising_weight > 0:
        logger.info(f"Denoising auxiliary: weight={args.denoising_weight}, mask_ratio={args.denoise_mask_ratio}")
    if args.diversity_weight > 0:
        logger.info(f"Diversity auxiliary: weight={args.diversity_weight}")
    if args.progressive_step_weight > 0:
        logger.info(f"Progressive step supervision: weight={args.progressive_step_weight}")
    if args.fresh_final:
        logger.info("Fresh final: all K positions use current hidden at final step")
    if args.soft_denoising:
        logger.info(f"Soft-token multi-step denoising: temperature={args.soft_temperature}")
    if args.corruption_rate > 0:
        logger.info(f"Corruption augmentation: max rate={args.corruption_rate}")
    if args.disable_hidden_hook:
        logger.info("Hidden hook disabled: using output_hidden_states=True")
    if args.debug_dense_metrics:
        logger.info("Dense debug metrics enabled")
    if args.debug_compare_hidden_once:
        logger.info("Hidden-hook verification enabled for first batch")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {n_trainable:,} / {n_total:,} params "
                f"({100 * n_trainable / n_total:.2f}%)")

    ret_tokenizer = RetrievalTokenizer(model)

    dataset = MSMARCOTrainDataset(
        data_source=args.train_data,
        n_negatives=args.n_negatives,
        max_examples=args.max_train_examples,
        seed=args.data_seed,
        split='train',
        ret_tokenizer=ret_tokenizer,
    )

    eval_dataset = None
    if args.eval_steps is not None:
        eval_source = args.eval_data or args.train_data
        eval_dataset = MSMARCOTrainDataset(
            data_source=eval_source,
            n_negatives=args.n_negatives,
            max_examples=args.max_eval_examples,
            seed=args.data_seed,
            split=args.eval_split,
            offset=args.eval_offset,
            ret_tokenizer=ret_tokenizer,
        )
        logger.info(f"Eval dataset: {len(eval_dataset)} examples "
                    f"from {eval_source} ({args.eval_split})")

    collator = RetrievalCollator(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,                                 # explicit (HF default)
        dataloader_prefetch_factor=(args.dataloader_prefetch_factor
                                    if args.dataloader_num_workers > 0 else None),
        dataloader_persistent_workers=(args.dataloader_persistent_workers
                                       and args.dataloader_num_workers > 0),
        deepspeed=args.deepspeed,
        optim='adamw_torch_fused',
        report_to='wandb',
        remove_unused_columns=False,
        seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=args.group_by_length,
        length_column_name='length',
        eval_strategy='steps' if eval_dataset is not None else 'no',
        eval_steps=args.eval_steps,
    )

    trainer = RetrievalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # Auto-detect checkpoint to resume from
    resume = args.resume_from_checkpoint
    if resume is None:
        resume = get_last_checkpoint(args.output_dir)
        if resume is not None:
            logger.info(f"Resuming from checkpoint: {resume}")
        else:
            logger.info("No checkpoint found, starting from scratch.")

    # --resume_weights_only: load model weights but skip DeepSpeed optimizer states.
    # Needed when resuming with a different GPU count (ZeRO world-size mismatch).
    if args.resume_weights_only and resume is not None:
        logger.info(f"Loading weights only from {resume} (skipping optimizer states)")
        import glob
        # With PEFT (LoRA), adapter weights are in adapter_model.bin / .safetensors
        adapter_bin = os.path.join(resume, 'adapter_model.bin')
        adapter_safe = os.path.join(resume, 'adapter_model.safetensors')
        model_bin = os.path.join(resume, 'pytorch_model.bin')
        if os.path.exists(adapter_safe) or os.path.exists(adapter_bin):
            model.backbone.load_adapter(resume, adapter_name='default')
            logger.info("Loaded PEFT adapter weights")
        elif os.path.exists(model_bin):
            state = torch.load(model_bin, map_location='cpu')
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info(f"Loaded pytorch_model.bin "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            shards = sorted(glob.glob(os.path.join(resume, 'pytorch_model-*.bin')))
            if shards:
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, resume, strict=False)
                logger.info(f"Loaded {len(shards)} sharded weight files")
            else:
                logger.warning(f"No model weight files found in {resume} — "
                               f"starting from scratch")
        resume = None  # Tell trainer not to load DeepSpeed optimizer states

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume)

    logger.info(f"Saving to {args.output_dir}")
    if args.deepspeed:
        # ZeRO-3 shards params across GPUs; trainer.save_model() gathers them correctly.
        trainer.save_model(args.output_dir)
        if trainer.is_world_process_zero():
            model.save(args.output_dir)
    else:
        model.save(args.output_dir)
    logger.info("Done.")


if __name__ == '__main__':
    main()
