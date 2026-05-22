#!/usr/bin/env python3
"""
Train AR (Llama / Qwen) retriever on MS MARCO.

Supports:
  K=1 (single-representation): hidden state at last " token — dense only or dense+sparse.
  K>1, causal (multi-representation): autoregressive PromptReps-style generation.
  K>1, bidirectional: PromptReps-style single-pass readout block
    [", pool_1, ..., pool_{K-1}] with causal masking disabled.

Loss matches TrainableDiffusionRetriever for fair comparison.

Usage:
    torchrun --nproc_per_node 4 scripts/train_promptreps.py \
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --model_type llama \
        --query_prompt  prompts/default/query_prompt.yaml \
        --passage_prompt prompts/default/passage_prompt.yaml \
        --n_pooled_tokens 1 \
        --output_dir models/llama_k1 \
        --lora_rank 64 --lora_alpha 64

    torchrun --nproc_per_node 4 scripts/train_promptreps.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --model_type qwen \
        --query_prompt  prompts/default/query_prompt.yaml \
        --passage_prompt prompts/default/passage_prompt.yaml \
        --n_pooled_tokens 2 \
        --output_dir models/qwen_k2 \
        --lora_rank 64 --lora_alpha 64

    torchrun --nproc_per_node 4 scripts/train_promptreps.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --model_type qwen \
        --query_prompt  prompts/default/query_prompt.yaml \
        --passage_prompt prompts/default/passage_prompt.yaml \
        --n_pooled_tokens 4 \
        --bidirectional \
        --output_dir models/qwen_k4_bidir \
        --lora_rank 64 --lora_alpha 64
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
# CPU-only tokenizer for DataLoader workers
# ──────────────────────────────────────────────────────────────────────────────

class ARRetrievalTokenizer:
    """Picklable tokenizer extracted from TrainableARRetriever for worker use."""

    def __init__(self, model):
        self.tokenizer = model.tokenizer
        self.query_prefix_ids = list(model._query_prefix_ids)
        self.query_suffix_ids = list(model._query_suffix_ids)
        self.passage_prefix_ids = list(model._passage_prefix_ids)
        self.passage_suffix_ids = list(model._passage_suffix_ids)
        self.pool_token_id = model.pool_token_id
        self.n_pooled_tokens = model.n_pooled_tokens
        self.bidirectional = model.bidirectional
        self.query_max_length = getattr(model, 'query_max_length', model.max_length)
        self.passage_max_length = getattr(model, 'passage_max_length', model.max_length)
        self.pad_token_id = (model.tokenizer.pad_token_id
                             or model.tokenizer.eos_token_id or 0)
        self.sparse_weight = model.sparse_weight

    def __call__(self, text: str, is_query: bool) -> Dict:
        prefix_ids = self.query_prefix_ids if is_query else self.passage_prefix_ids
        suffix_ids = self.query_suffix_ids if is_query else self.passage_suffix_ids
        K = self.n_pooled_tokens
        # Bidirectional K>1: pre-append K-1 pool tokens so the final K
        # positions are [", pool_1, ..., pool_{K-1}] in one full-attention pass.
        # Causal K>1: no pool tokens — generated autoregressively in encode()
        pool_tail = ([self.pool_token_id] * (K - 1)
                     if (self.bidirectional and K > 1) else [])
        max_len = self.query_max_length if is_query else self.passage_max_length
        # PromptReps semantic: `max_len` bounds the TEXT ONLY. Prompt prefix/suffix
        # and pool tokens are appended on top. See PromptReps/dataset.py. So
        # --query_max_len 32 means 32 tokens of raw query text.
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
        ids = prefix_ids + enc['input_ids'][0] + suffix_ids + pool_tail
        result = {'input_ids': ids, 'attention_mask': [1] * len(ids)}
        if self.sparse_weight > 0:
            if not hasattr(self, '_get_content_ids'):
                from models.sparse_utils import get_content_token_ids
                self._get_content_ids = get_content_token_ids
            result['content_ids'] = list(self._get_content_ids([text], self.tokenizer)[0])
        return result

    def from_pretokenized(self, text_tokens, content_ids, is_query: bool) -> Dict:
        """Build the input dict from already-tokenized text tokens + sparse
        content_ids (precomputed offline by scripts/preprocess_msmarco_aug.py).
        Mirrors the layout that __call__ produces, just skipping the
        tokenizer + content-extraction calls."""
        prefix_ids = self.query_prefix_ids if is_query else self.passage_prefix_ids
        suffix_ids = self.query_suffix_ids if is_query else self.passage_suffix_ids
        K = self.n_pooled_tokens
        pool_tail = ([self.pool_token_id] * (K - 1)
                     if (self.bidirectional and K > 1) else [])
        ids = list(prefix_ids) + list(text_tokens) + list(suffix_ids) + pool_tail
        result = {'input_ids': ids, 'attention_mask': [1] * len(ids)}
        if self.sparse_weight > 0:
            result['content_ids'] = list(content_ids)
        return result


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from models.promptreps_trainable import TrainableARRetriever

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset  (identical to train_diffretriever.py)
# ──────────────────────────────────────────────────────────────────────────────

class MSMARCOTrainDataset(Dataset):
    def __init__(
        self,
        data_source: str,
        n_negatives: int = 7,
        max_examples: int = 0,
        seed: int = 42,
        split: str = 'train',
        offset: int = 0,
        ret_tokenizer: Optional[ARRetrievalTokenizer] = None,
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

        # Detect whether the dataset was preprocessed by
        # scripts/preprocess_msmarco_aug.py (has `query_tokens` column instead
        # of raw `query` text).  If yes, __getitem__ will skip tokenization.
        cols = (self.data.column_names if hasattr(self.data, 'column_names')
                else (list(self.data[0].keys()) if self.data else []))
        self.pretokenized = 'query_tokens' in cols
        if self.pretokenized:
            logger.info("Detected pre-tokenized dataset → skipping per-call "
                        "tokenization (faster DataLoader, exact lengths).")

        logger.info(f"Loaded {len(self.data)} examples (split={split}, offset={offset})")

        # Per-example length for length-grouped batch sampling.  Pretokenized
        # datasets ship an exact `length` column; fall back to char-count proxy
        # otherwise.  Stored as `self.lengths`; HF Trainer's LengthGroupedSampler
        # reads this via our RetrievalTrainer._get_train_sampler override.
        self.lengths = self._compute_lengths()
        logger.info(f"Length stats: min={min(self.lengths)} "
                    f"median={sorted(self.lengths)[len(self.lengths)//2]} "
                    f"max={max(self.lengths)}"
                    f"{'  (exact tokens)' if self.pretokenized else '  (char-count proxy)'}")

    def _compute_lengths(self) -> List[int]:
        """Length per example.  Pretokenized: read 'length' column.  Otherwise
        fall back to char-count proxy (rough but correlated with tokens)."""
        # Pretokenized fast path
        if self.pretokenized and hasattr(self.data, 'column_names') \
                and 'length' in self.data.column_names:
            return list(self.data['length'])
        if self.pretokenized:
            return [int(item.get('length', 0)) for item in self.data]

        # Char-count proxy
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

    def __getitem__(self, idx: int) -> Dict:
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
                # Pretokenization should always have positives; defensive only.
                p_pos = {'tokens': [], 'content_ids': []}
            else:
                p_pos = pos_list[(item_seed + self.epoch) % len(pos_list)]

            neg_list = list(item.get('negative_passages') or [])
            n_neg = self.n_negatives
            sampled_negs: List = []
            if neg_list:
                shuffled = list(neg_list)
                rng.shuffle(shuffled)
                if len(shuffled) >= n_neg:
                    offset = (self.epoch * n_neg) % len(shuffled)
                    sampled_negs = (shuffled + shuffled)[offset: offset + n_neg]
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
            # Raw-text fallback shouldn't be hit when pretokenized; bail.
            raise RuntimeError("ret_tokenizer is required for pretokenized data")

        # ── Raw-text path (legacy) ─────────────────────────────────────────
        query = item['query']
        pos_list = item.get('positive_passages', [])
        if not pos_list:
            pos_text = item.get('positive', '')
        else:
            pos = pos_list[(item_seed + self.epoch) % len(pos_list)]
            title = pos.get('title', '')
            text = pos.get('text', pos.get('contents', ''))
            pos_text = f"{title} {text}".strip() if title else text

        neg_list = list(item.get('negative_passages') or [])
        n_neg = self.n_negatives
        sampled_negs: List = []
        if neg_list:
            shuffled = list(neg_list)
            rng.shuffle(shuffled)
            if len(shuffled) >= n_neg:
                offset = (self.epoch * n_neg) % len(shuffled)
                sampled_negs = (shuffled + shuffled)[offset: offset + n_neg]
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
            return {
                'query': self.ret_tokenizer(query, is_query=True),
                'passages': [self.ret_tokenizer(p, is_query=False)
                             for p in [pos_text] + neg_texts],
            }
        return {'query': query, 'passages': [pos_text] + neg_texts}


# ──────────────────────────────────────────────────────────────────────────────
# Collator
# ──────────────────────────────────────────────────────────────────────────────

class ARRetrievalCollator:
    def __init__(self, model: TrainableARRetriever, pad_token_id: Optional[int] = None):
        self.model = model
        self.pad_token_id = (pad_token_id
                             or model.tokenizer.pad_token_id
                             or model.tokenizer.eos_token_id or 0)
        self.padding_side = model.tokenizer.padding_side

    def _pad(self, encs: List[Dict], pad_to_multiple_of: int = 8):
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
            if self.padding_side == 'left':
                # Left-pad: content right-aligned, pad tokens at the start
                ids[i, max_len - l:] = torch.tensor(e['input_ids'], dtype=torch.long)
                mask[i, max_len - l:] = torch.tensor(e['attention_mask'], dtype=torch.long)
            else:
                ids[i, :l] = torch.tensor(e['input_ids'], dtype=torch.long)
                mask[i, :l] = torch.tensor(e['attention_mask'], dtype=torch.long)
        return ids, mask

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(batch[0]['query'], dict):
            q_encs = [item['query'] for item in batch]
            p_encs = [p for item in batch for p in item['passages']]
            q_ids, q_mask = self._pad(q_encs)
            p_ids, p_mask = self._pad(p_encs)
        else:
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
    def _get_train_sampler(self, train_dataset=None):
        """Override stock Trainer sampler logic.  When `group_by_length=True`,
        Trainer's stock path only reads lengths from a `datasets.Dataset`'s
        `length_column_name` column — but our `MSMARCOTrainDataset` is a custom
        torch Dataset wrapping (sometimes) an HF Dataset.  We expose lengths via
        `train_dataset.lengths` and feed them straight to LengthGroupedSampler.
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
    parser = argparse.ArgumentParser(description='Train AR (Llama/Qwen) retriever')

    # Model
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_type', required=True, choices=['llama', 'qwen', 'qwen25'])
    parser.add_argument('--query_prompt', required=True)
    parser.add_argument('--passage_prompt', required=True)

    # Encoding
    parser.add_argument('--n_pooled_tokens', type=int, default=1,
                        help='Number of retrieval readout tokens K')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Disable causal masking for an encoder-style control. '
                             'For K>1 this yields a single-pass readout block '
                             '[quote, pool_1, ..., pool_{K-1}] rather than '
                             'autoregressive generation. This is LLM2Vec-inspired '
                             'attention, not the full LLM2Vec training recipe.')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Legacy single max length (overridden by query/passage_max_length)')
    parser.add_argument('--query_max_length', type=int, default=None)
    parser.add_argument('--passage_max_length', type=int, default=None)
    parser.add_argument('--normalize', action='store_true', default=True)

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=0)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Data
    parser.add_argument('--train_data', default='Tevatron/msmarco-passage-aug')
    parser.add_argument('--n_negatives', type=int, default=7)
    parser.add_argument('--max_train_examples', type=int, default=0)
    parser.add_argument('--data_seed', type=int, default=42)

    # Eval
    parser.add_argument('--eval_data', default=None)
    parser.add_argument('--eval_split', default='train')
    parser.add_argument('--eval_offset', type=int, default=500000)
    parser.add_argument('--max_eval_examples', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=None)

    # Loss
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--dense_weight', type=float, default=1.0,
                        help='Weight for dense retrieval loss (0 = sparse-only training)')
    parser.add_argument('--sparse_weight', type=float, default=1.0)

    # Training
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--per_device_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_scheduler_type', default='cosine')
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--dataloader_num_workers', type=int, default=8,
                        help='Tokenisation + content_token_ids run in workers. '
                             '8/rank × 4 GPUs = 32 workers — plenty given 100%% GPU util '
                             'means the data path is not the bottleneck.')
    parser.add_argument('--dataloader_prefetch_factor', type=int, default=4)
    parser.add_argument('--dataloader_persistent_workers', action='store_true', default=True)
    parser.add_argument('--deepspeed', default=None)
    parser.add_argument('--no_gradient_checkpointing', action='store_true')
    parser.add_argument('--group_by_length', action='store_true', default=False,
                        help='Group similar-length sequences into the same batch '
                             'to reduce padding waste. ~5-15%% speedup, no recipe '
                             'change (loss math identical).  Apply uniformly across '
                             'all runs in a comparison group.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_from_checkpoint', default=None)
    parser.add_argument('--resume_weights_only', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    # Resolve separate query/passage max lengths (fall back to --max_length)
    q_max = args.query_max_length or args.max_length
    p_max = args.passage_max_length or args.max_length

    logger.info(f"Building model: {args.model_name} ({args.model_type}) K={args.n_pooled_tokens}")
    model = TrainableARRetriever.from_pretrained(
        model_name=args.model_name,
        model_type=args.model_type,
        query_prompt=args.query_prompt,
        passage_prompt=args.passage_prompt,
        max_length=p_max,
        n_pooled_tokens=args.n_pooled_tokens,
        temperature=args.temperature,
        sparse_weight=args.sparse_weight,
        normalize=args.normalize,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bidirectional=args.bidirectional,
    )
    model.query_max_length = q_max
    model.passage_max_length = p_max
    model.dense_weight = args.dense_weight

    if args.dense_weight != 1.0:
        logger.info(f"Dense loss weight: {args.dense_weight}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {n_trainable:,} / {n_total:,} params "
                f"({100 * n_trainable / n_total:.2f}%)")

    ret_tokenizer = ARRetrievalTokenizer(model)

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
        logger.info(f"Eval dataset: {len(eval_dataset)} examples")

    collator = ARRetrievalCollator(model)

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
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=(args.dataloader_prefetch_factor
                                    if args.dataloader_num_workers > 0 else None),
        dataloader_persistent_workers=(args.dataloader_persistent_workers
                                       and args.dataloader_num_workers > 0),
        deepspeed=args.deepspeed,
        optim='adamw_torch_fused',
        report_to='wandb',
        remove_unused_columns=False,
        group_by_length=args.group_by_length,
        length_column_name='length',
        seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    resume = args.resume_from_checkpoint
    if resume is None:
        resume = get_last_checkpoint(args.output_dir)
        if resume is not None:
            logger.info(f"Resuming from checkpoint: {resume}")
        else:
            logger.info("No checkpoint found, starting from scratch.")

    if args.resume_weights_only and resume is not None:
        logger.info(f"Loading weights only from {resume} (skipping optimizer states)")
        import glob
        adapter_safe = os.path.join(resume, 'adapter_model.safetensors')
        adapter_bin = os.path.join(resume, 'adapter_model.bin')
        if os.path.exists(adapter_safe) or os.path.exists(adapter_bin):
            model.backbone.load_adapter(resume, adapter_name='default')
            logger.info("Loaded PEFT adapter weights")
        else:
            shards = sorted(glob.glob(os.path.join(resume, 'pytorch_model*.bin')))
            if shards:
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, resume, strict=False)
            else:
                logger.warning(f"No weight files found in {resume} — starting from scratch")
        resume = None

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume)

    logger.info(f"Saving to {args.output_dir}")
    if args.deepspeed:
        trainer.save_model(args.output_dir)
        if trainer.is_world_process_zero():
            model.save(args.output_dir)
    else:
        model.save(args.output_dir)
    logger.info("Done.")


if __name__ == '__main__':
    main()
