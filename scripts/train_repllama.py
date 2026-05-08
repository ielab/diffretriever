#!/usr/bin/env python3
"""Fine-tune LLaMA as a RepLLaMA-style retriever (Ma et al. 2024).

Mirrors scripts/train_diffembed.py exactly — same data path, same negative
sampling, same lr, same lora, same warmup ratio, same epochs, same seed.
The only differences are:
  * backbone is causal LLaMA (not bidirectional Dream/LLaDA)
  * pool is last-token (EOS) instead of mean
Loss (InfoNCE @ τ=0.02) is identical.

Usage:
  torchrun --nproc_per_node=4 scripts/train_repllama.py \\
      --backbone meta-llama/Meta-Llama-3-8B-Instruct \\
      --train_data data/msmarco-passage-aug \\
      --output_dir models_new/repllama_llama-lora16 \\
      --lora_rank 16 --learning_rate 1e-4 --epochs 1 \\
      --deepspeed configs/ds_zero2.json
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.repllama_retriever import RepLLaMARetriever      # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────
class TripletDataset(Dataset):
    """Same wrapper as train_diffembed.TripletDataset — yields one query +
    (1 pos + n_neg sampled negs) per __getitem__.  Identical for parity."""

    def __init__(self, hf_dataset, n_negatives: int = 15, seed: int = 0):
        from datasets import load_from_disk
        if isinstance(hf_dataset, (str, Path)):
            ds = load_from_disk(str(hf_dataset))
            if hasattr(ds, 'keys'):
                ds = ds[list(ds.keys())[0]]
            self.ds = ds
        else:
            self.ds = hf_dataset
        self.n_neg = n_negatives
        self.rng = random.Random(seed)

        cols = set(self.ds.column_names)
        self.q_col = next((c for c in ('query', 'q', 'question') if c in cols), None)
        self.pos_col = next((c for c in ('positive_passages', 'positives') if c in cols), None)
        self.neg_col = next((c for c in ('negative_passages', 'negatives') if c in cols), None)
        if not (self.q_col and self.pos_col):
            raise ValueError(f"Need query + positive columns, got {sorted(cols)}")

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _text_of(p):
        if isinstance(p, dict):
            for k in ('text', 'passage', 'body', 'content'):
                if k in p:
                    return p[k]
            for k in ('title',):
                if k in p:
                    return p[k]
        return str(p)

    def __getitem__(self, i):
        row = self.ds[i]
        q = row[self.q_col] if isinstance(row[self.q_col], str) \
            else row[self.q_col].get('text', str(row[self.q_col]))
        pos = row[self.pos_col]
        if isinstance(pos, list):
            pos_text = self._text_of(self.rng.choice(pos))
        else:
            pos_text = self._text_of(pos)

        negs: List[str] = []
        if self.neg_col:
            raw_negs = row[self.neg_col]
            if isinstance(raw_negs, list) and raw_negs:
                pool = [self._text_of(n) for n in raw_negs]
                self.rng.shuffle(pool)
                negs = pool[:self.n_neg]
        while len(negs) < self.n_neg:
            j = self.rng.randint(0, len(self.ds) - 1)
            other = self.ds[j][self.pos_col]
            other_text = self._text_of(other[0] if isinstance(other, list) else other)
            negs.append(other_text)

        return {'query': q, 'passages': [pos_text] + negs}


# ── Collator ─────────────────────────────────────────────────────────────────
class RepLLaMACollator:
    """Right-pad tokenize on the fly + append EOS via the model's tokenize."""

    def __init__(self, model: RepLLaMARetriever, query_max_len: int = 32,
                 passage_max_len: int = 156):
        self.model = model
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

    def __call__(self, batch):
        queries = [b['query'] for b in batch]
        passages = [p for b in batch for p in b['passages']]
        n_pass_per_q = len(batch[0]['passages'])

        # Use the model's tokenize (which appends EOS).  We temporarily swap
        # max_length per side to respect the per-side caps.
        orig_max = self.model.max_length
        self.model.max_length = self.query_max_len
        q_ids, q_mask = self.model.tokenize(queries)
        self.model.max_length = self.passage_max_len
        p_ids, p_mask = self.model.tokenize(passages)
        self.model.max_length = orig_max

        labels = torch.arange(len(batch), dtype=torch.long) * n_pass_per_q
        return {
            'query_input_ids': q_ids,
            'query_attention_mask': q_mask,
            'passage_input_ids': p_ids,
            'passage_attention_mask': p_mask,
            'labels': labels,
        }


# ── Trainer ──────────────────────────────────────────────────────────────────
class RepLLaMATrainer(Trainer):
    """HF Trainer wrapper — calls model.forward_train() with InfoNCE loss.

    Unwraps DataParallel / DDP / DeepSpeedEngine so the custom forward_train
    method is reachable.  ``nn.DataParallel`` only proxies ``forward()``;
    DDP raises on non-forward calls; DeepSpeedEngine works via ``__getattr__``
    delegation but unwrapping is uniformly safe.
    """

    @staticmethod
    def _unwrap(m):
        # Handles .module (DDP / DataParallel / DeepSpeedEngine) chains
        while hasattr(m, 'module') and m.module is not m:
            m = m.module
        return m

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        inner = self._unwrap(model)
        out = inner.forward_train(
            query_input_ids=inputs['query_input_ids'],
            query_attention_mask=inputs['query_attention_mask'],
            passage_input_ids=inputs['passage_input_ids'],
            passage_attention_mask=inputs['passage_attention_mask'],
            labels=inputs['labels'],
        )
        return (out['loss'], out) if return_outputs else out['loss']


# ── CLI / main ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', default='meta-llama/Meta-Llama-3-8B-Instruct',
                   help='HF model id')
    p.add_argument('--train_data', required=True, type=Path,
                   help='HF dataset dir with msmarco-passage-aug-style triples')

    p.add_argument('--output_dir', required=True, type=Path)
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=64)
    p.add_argument('--lora_dropout', type=float, default=0.05)

    p.add_argument('--n_negatives', type=int, default=15)
    p.add_argument('--query_max_len', type=int, default=32)
    p.add_argument('--passage_max_len', type=int, default=156)

    # Side-marker prefixes (RepLLaMA paper, Ma 2024).  Defaults match the
    # paper recipe: 'query: ' and 'passage: '.  These strings are part of
    # the input — model learns to use them as side markers, which transfers
    # better to OOD retrieval (BEIR).
    p.add_argument('--query_prefix', default='query: ')
    p.add_argument('--passage_prefix', default='passage: ')

    p.add_argument('--per_device_batch_size', type=int, default=8)
    p.add_argument('--total_batch_size', type=int, default=128,
                   help='Effective batch (= per_device × n_gpus × grad_accum).')
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--warmup_steps', type=int, default=0,
                   help='Fixed warmup steps. If 0 and --warmup_ratio>0, ratio is used instead.')
    p.add_argument('--warmup_ratio', type=float, default=0.06,
                   help='Warmup as fraction of total steps (matches train.sh: 0.06).')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--max_steps', type=int, default=-1)
    p.add_argument('--save_steps', type=int, default=2000)
    p.add_argument('--logging_steps', type=int, default=50)

    p.add_argument('--deepspeed', default=None, help='Path to DeepSpeed JSON config')
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--bf16', action='store_true', default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume_from_checkpoint', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n_gpus = max(1, torch.cuda.device_count())
    grad_accum = max(1, args.total_batch_size // (args.per_device_batch_size * n_gpus))
    logger.info(f"GPUs={n_gpus}, per_device_bs={args.per_device_batch_size}, "
                f"grad_accum={grad_accum} → effective_bs={args.per_device_batch_size * n_gpus * grad_accum}")

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info(f"Building RepLLaMARetriever (backbone={args.backbone}, "
                f"lora_rank={args.lora_rank})")
    model = RepLLaMARetriever(
        model_name=args.backbone,
        max_length=max(args.query_max_len, args.passage_max_len),
        normalize=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        attn_implementation='flash_attention_2',
        device_map=None,                              # let Trainer place
        gradient_checkpointing=args.gradient_checkpointing,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info(f"Loading triples from {args.train_data}")
    train_ds = TripletDataset(args.train_data, n_negatives=args.n_negatives, seed=args.seed)
    logger.info(f"  {len(train_ds):,} training examples")

    collator = RepLLaMACollator(
        model,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio if args.warmup_steps == 0 else 0.0,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to='wandb',
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = RepLLaMATrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    # Auto-resume
    resume = args.resume_from_checkpoint
    if resume is None:
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(str(args.output_dir))
        if resume:
            logger.info(f"Auto-resuming from {resume}")

    trainer.train(resume_from_checkpoint=resume)

    # ── Save ──────────────────────────────────────────────────────────────
    logger.info(f"Saving RepLLaMA retriever to {args.output_dir}")
    if hasattr(trainer, 'is_world_process_zero') and not trainer.is_world_process_zero():
        return
    model.save(args.output_dir)


if __name__ == '__main__':
    main()
