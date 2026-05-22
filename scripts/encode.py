#!/usr/bin/env python3
"""
Encode queries/passages using the PromptReps-style retrieval prompt with
K representations per text (single-representation when K=1, multi-representation
when K>1).

Supported backbones (via --model_type):
    Diffusion: dream, llada1, llada15, llada2
    AR:        llama, qwen25, qwen3
    Trained:   trained_diff, trained_ar (pair with --model_dir)

The default --encode_type is 'all_steps' (writes repr_hidden + sparse
indices/values in one shard). 'dense' and 'sparse' produce the legacy
standalone outputs; both go through save_shard which writes
.safetensors.zst by default (falls back to .safetensors, then .pt).

Usage:
    # Zero-shot diffusion — multi-representation, K=4 masked positions
    python scripts/encode.py \\
        --model_type dream --n_gen_tokens 4 --num_denoise_steps 1 \\
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \\
        --input_file data/msmarco/queries.dev.jsonl \\
        --output_dir embeddings/msmarco/dream_few_k4_s1/queries \\
        --is_query \\
        --query_prompt prompts/default/query_prompt_few.yaml \\
        --passage_prompt prompts/default/passage_prompt_few.yaml

    # Zero-shot AR PromptReps — LLaMA-3 single-representation (K=1)
    python scripts/encode.py \\
        --model_type llama --num_pooled_tokens 0 \\
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \\
        --input_file data/msmarco/corpus.jsonl \\
        --output_dir embeddings/msmarco/llama_one_s1/corpus \\
        --query_prompt  prompts/default/query_prompt_one.yaml \\
        --passage_prompt prompts/default/passage_prompt_one.yaml \\
        --shard_id 0 --num_shards 350

    # Trained diffusion checkpoint
    python scripts/encode.py \\
        --model_type trained_diff --model_dir models_new/dream_few-k4-s1-lora16-sp1.0 \\
        --input_file data/msmarco/queries.dev.jsonl \\
        --output_dir embeddings/msmarco/trained_diff_dream_few_k4_s1-lora16-sp1.0/queries-dev \\
        --is_query
"""

import sys
import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

sys.path.insert(0, str(project_root / 'scripts'))
from shard_io import save_shard, pool_sparse_across_k

from models.promptreps import PromptRepsRetriever
from models.diffretriever_llada import LLaDA2Retriever
from models.diffretriever_dream import DreamRetriever
from models.block_schedule import BlockSchedule
from models.diffretriever_trainable import TrainableDiffusionRetriever
from models.promptreps_trainable import TrainableARRetriever
from models.sparse_utils import get_content_token_ids, filter_sparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_item(item, idx):
    """Extract id and text from a parsed JSONL item dict."""
    doc_id = str(item.get('id', item.get('_id', item.get('docid', item.get('query_id', idx)))))
    text = item.get('text', item.get('contents', item.get('query', '')))
    title = item.get('title', '')
    if title:
        text = title + '. ' + text if text else title
    return doc_id, text


def load_texts(input_file: str, shard_id: int = -1, num_shards: int = 1):
    """Load texts and ids from JSONL file.

    Handles MS MARCO (id/contents/query), BEIR (_id/title+text), and TREC formats.

    When shard_id >= 0, only the shard's lines are JSON-parsed (two-pass: fast
    line count then targeted read). This avoids loading the entire file on every
    shard worker — critical for large corpora like HotpotQA (5M+ passages).
    """
    if shard_id < 0:
        ids, texts = [], []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                doc_id, text = _parse_item(json.loads(line), idx)
                ids.append(doc_id)
                texts.append(text)
        return ids, texts

    # Pass 1: count total lines (no JSON parsing — just counting newlines)
    with open(input_file, 'rb') as f:
        total = sum(1 for _ in f)

    chunk = math.ceil(total / num_shards)
    start = shard_id * chunk
    end = min(start + chunk, total)

    # Pass 2: parse only lines in [start, end)
    ids, texts = [], []
    with open(input_file) as f:
        for line_num, line in enumerate(f):
            if line_num < start:
                continue
            if line_num >= end:
                break
            doc_id, text = _parse_item(json.loads(line), line_num)
            ids.append(doc_id)
            texts.append(text)
    return ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices=['llama', 'qwen', 'qwen25', 'qwen3', 'llada1', 'llada15', 'llada2', 'llada21', 'dream', 'trained_diff', 'trained_ar', 'diffembed_dream', 'diffembed_llada1', 'diffembed_llada15', 'diffembed_llada2', 'repllama'],
                        help='llama/qwen/qwen25/qwen3 = AR PromptReps, llada1/llada15/llada2/dream = diffusion, trained_diff = fine-tuned diffusion retriever, trained_ar = fine-tuned AR retriever')
    parser.add_argument('--model_name_or_path', default=None,
                        help='HF model name or path (not needed for trained_diff/trained_ar; use --model_dir)')
    parser.add_argument('--model_dir', default=None,
                        help='Path to fine-tuned checkpoint directory (trained_diff or trained_ar type)')
    parser.add_argument('--backbone_model_type', default=None, choices=['dream', 'llada1', 'llada15', 'llada2'],
                        help='Backbone type for trained_diff checkpoints without retriever_config.json')
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--is_query', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--query_max_len', type=int, default=512,
                        help='Max sequence length for queries (default: 512, matching PromptReps).')
    parser.add_argument('--passage_max_len', type=int, default=512,
                        help='Max sequence length for passages (default: 512, matching PromptReps).')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Override both query_max_len and passage_max_len with a single value.')

    # Encoding type
    parser.add_argument('--encode_type', choices=['all', 'dense', 'sparse', 'all_steps'], default='all',
                        help='What to encode: all (default), dense only, sparse only, '
                             'or all_steps (saves repr_hidden [B,n_gen,H] at each token decode step + '
                             'per-position sparse — one run covers all K/S combos at eval time)')

    # Prompt arguments — YAML files with chat template (one per query/passage side)
    parser.add_argument('--query_prompt', default=None,
                        help='Path to YAML prompt file for queries (not needed for trained_diff).')
    parser.add_argument('--passage_prompt', default=None,
                        help='Path to YAML prompt file for passages (not needed for trained_diff).')

    # Sparse embedding options
    parser.add_argument('--sparse_topk', type=int, default=128,
                        help='Keep top-k logit entries per sparse vector during encoding (0=skip sparse). '
                             'Use a large value (e.g. 5000) with --sparse_save_min to capture all non-zeros.')
    parser.add_argument('--sparse_save_min', type=int, default=0,
                        help='At save time, trim trailing-zero columns but keep at least this many. '
                             '0 = no trimming (save full sparse_topk). '
                             'E.g. --sparse_topk 5000 --sparse_save_min 1000 saves all non-zero entries, '
                             'minimum 1000 columns.')

    # Sharding (for parallel corpus encoding)
    parser.add_argument('--shard_id', type=int, default=-1,
                        help='Shard index (0-based). -1 = no sharding (default)')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Total number of shards')

    # AR-specific
    parser.add_argument('--num_pooled_tokens', type=int, default=0,
                        help='AR PromptReps: 0=single-representation (one token), '
                             '>0=multi-representation (autoregressively generate up '
                             'to this many tokens; the actual K is the count produced '
                             'before the closing quote)')
    parser.add_argument('--normalize', action='store_true')

    # Attention backend.  Default is flash_attention_2 (~1.5-3x faster than
    # eager on long sequences).  Switch to eager / sdpa if flash is broken
    # or wedged on a particular cluster.
    parser.add_argument('--attn_implementation', type=str,
                        default='flash_attention_2',
                        choices=('flash_attention_2', 'flash_attention_3',
                                 'sdpa', 'eager'),
                        help='Attention backend passed to from_pretrained. '
                             'Default flash_attention_2; try eager or sdpa '
                             'if flash misbehaves on this GPU.')

    # LLaDA 2-specific
    parser.add_argument('--num_denoise_steps', type=int, default=1,
                        help='LLaDA 2: denoising steps (1=single pass, >1=iterative)')
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--no_quotation_token', action='store_false', dest='use_quotation_token',
                        help='Use [MASK] hidden state instead of the " token for dense (n_repr=1 only). '
                             'Default is quotation token.')
    parser.add_argument('--n_gen_tokens', type=int, default=0,
                        help='K: number of [MASK] generation tokens. Controls both the '
                             'prompt text ("Use K words") and the MASK count. 0=disabled (default).')
    parser.add_argument('--filter_structural', action='store_true',
                        help='Zero out repr positions that decoded to quote-containing tokens '
                             '(structural noise — model generating closing sequence early).')
    args = parser.parse_args()

    # --max_length overrides both sides; otherwise use per-side defaults
    if args.max_length is None:
        args.max_length = args.query_max_len if args.is_query else args.passage_max_len

    os.makedirs(args.output_dir, exist_ok=True)

    # Output base names (no extension — save_shard picks .safetensors.zst when
    # `zstandard` is installed, falls back to .safetensors, or .pt as last resort).
    if args.shard_id >= 0:
        dense_base = f'shard_{args.shard_id}'
        sparse_base = f'sparse_shard_{args.shard_id}'
        all_steps_base = f'all_steps_shard_{args.shard_id}'
    else:
        dense_base = 'embeddings'
        sparse_base = 'sparse'
        all_steps_base = 'all_steps_embeddings'

    dense_path = os.path.join(args.output_dir, dense_base)
    sparse_path = os.path.join(args.output_dir, sparse_base)
    all_steps_path = os.path.join(args.output_dir, all_steps_base)

    def _output_exists(base: str) -> bool:
        """True if an output for this base was written in any supported format."""
        for ext in ('.safetensors.zst', '.safetensors', '.pt'):
            if os.path.exists(base + ext):
                return True
        return False

    if args.encode_type == 'all_steps' and _output_exists(all_steps_path):
        logger.info("all_steps output already exists — skipping")
        return
    if args.encode_type == 'dense' and _output_exists(dense_path):
        logger.info(f"Dense output already exists (base={dense_path}) — skipping")
        return
    if args.encode_type == 'sparse' and _output_exists(sparse_path):
        logger.info(f"Sparse output already exists (base={sparse_path}) — skipping")
        return
    if (args.encode_type == 'all' and _output_exists(dense_path)
            and _output_exists(sparse_path)):
        logger.info("Both dense and sparse outputs already exist — skipping")
        return

    logger.info(f"Model type: {args.model_type}, Encode type: {args.encode_type}")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Is query: {args.is_query}")

    # Load data — shard-aware: only parses this shard's lines (avoids full file load per worker)
    all_ids, all_texts = load_texts(args.input_file, shard_id=args.shard_id, num_shards=args.num_shards)
    if args.shard_id >= 0:
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: loaded {len(all_texts)} items")
    else:
        logger.info(f"Loaded {len(all_texts)} texts")

    # Validate args
    if args.model_type in ('trained_diff', 'trained_ar'):
        if not args.model_dir:
            parser.error("--model_dir is required for model_type=trained_diff/trained_ar")
    else:
        if not args.model_name_or_path:
            parser.error("--model_name_or_path is required for this model_type")
        if not args.query_prompt or not args.passage_prompt:
            parser.error("--query_prompt and --passage_prompt are required for this model_type")

    # Build model
    if args.model_type == 'trained_diff':
        model = TrainableDiffusionRetriever.load(
            args.model_dir,
            model_type=getattr(args, 'backbone_model_type', None),
            query_prompt=args.query_prompt,
            passage_prompt=args.passage_prompt,
            n_gen_tokens=args.n_gen_tokens or 4,
            num_denoise_steps=args.num_denoise_steps,
        )
        model.max_length = args.max_length  # apply query/passage max len from CLI
        model.eval()
    elif args.model_type == 'trained_ar':
        # Detect AR model type from model_dir name or backbone_model_type arg
        _ar_type = getattr(args, 'backbone_model_type', None)
        if not _ar_type:
            # Check dir name; if checkpoint-N, check parent dir
            _bn = os.path.basename(os.path.normpath(args.model_dir))
            if _bn.startswith('checkpoint-'):
                _bn = os.path.basename(os.path.dirname(os.path.normpath(args.model_dir)))
            if 'llama' in _bn.lower(): _ar_type = 'llama'
            elif 'qwen' in _bn.lower(): _ar_type = 'qwen'
        model = TrainableARRetriever.load(
            args.model_dir,
            model_type=_ar_type,
            query_prompt=args.query_prompt,
            passage_prompt=args.passage_prompt,
            n_pooled_tokens=args.n_gen_tokens or 1,
        )
        model.max_length = args.max_length  # apply query/passage max len from CLI
        model.eval()
    elif args.model_type in ('llama', 'qwen', 'qwen25', 'qwen3'):
        model = PromptRepsRetriever(
            model_name=args.model_name_or_path,
            max_length=args.max_length,
            normalize=args.normalize,
            num_pooled_tokens=args.num_pooled_tokens,
            query_prompt=args.query_prompt,
            passage_prompt=args.passage_prompt,
            attn_implementation=args.attn_implementation,
        )
    elif args.model_type in ('llada1', 'llada15', 'llada2', 'llada21'):
        schedule = BlockSchedule(block_length=args.block_length)
        model = LLaDA2Retriever(
            model_name=args.model_name_or_path,
            max_length=args.max_length,
            normalize=args.normalize,
            num_denoise_steps=args.num_denoise_steps,
            block_schedule=schedule,
            query_prompt=args.query_prompt,
            passage_prompt=args.passage_prompt,
            use_quotation_token=args.use_quotation_token,
            n_gen_tokens=args.n_gen_tokens,
            filter_structural=args.filter_structural,
            attn_implementation=args.attn_implementation,
        )
    elif args.model_type.startswith('diffembed_'):
        # DiffEmbed-style (Zhang et al. 2025, arxiv 2505.15045):
        # raw text → bidirectional forward → mean-pool. K=1, no sparse, no prompts.
        # model_type = 'diffembed_dream' / 'diffembed_llada1' / 'diffembed_llada2' / 'diffembed_llada15'
        from models.diffembed import DiffEmbedRetriever
        backbone_type = args.model_type[len('diffembed_'):]   # 'dream' / 'llada1' / ...
        # Trained checkpoint?  --model_dir indicates a saved DiffEmbed dir.
        if args.model_dir:
            model = DiffEmbedRetriever.load(
                args.model_dir,
                model_type=backbone_type,
                max_length=args.max_length,
                attn_implementation=args.attn_implementation,
            )
        else:
            model = DiffEmbedRetriever(
                model_name=args.model_name_or_path,
                model_type=backbone_type,
                max_length=args.max_length,
                normalize=args.normalize,
                lora_rank=0,                         # zero-shot
                attn_implementation=args.attn_implementation,
            )
    elif args.model_type == 'repllama':
        # RepLLaMA (Ma et al. 2024, arxiv 2310.08319): causal LLaMA, EOS appended,
        # last-token (EOS) hidden state as the embedding.  Trained-only baseline —
        # the EOS pool is meaningless without contrastive fine-tuning, so we always
        # require --model_dir pointing at a fine-tuned checkpoint.
        from models.repllama import RepLLaMARetriever
        if not args.model_dir:
            raise ValueError(
                "--model_type repllama requires --model_dir PATH. "
                "RepLLaMA is trained-only (no zero-shot variant). "
                "Train one with: bash scripts/train_repllama.sh"
            )
        model = RepLLaMARetriever.load(
            args.model_dir,
            max_length=args.max_length,
            attn_implementation=args.attn_implementation,
        )
    else:  # dream
        model = DreamRetriever(
            model_name=args.model_name_or_path,
            max_length=args.max_length,
            normalize=args.normalize,
            num_denoise_steps=args.num_denoise_steps,
            query_prompt=args.query_prompt,
            passage_prompt=args.passage_prompt,
            use_quotation_token=args.use_quotation_token,
            n_gen_tokens=args.n_gen_tokens,
            filter_structural=args.filter_structural,
            attn_implementation=args.attn_implementation,
        )

    # Model is already placed on GPU(s) via device_map='auto' in the retriever.
    # Do NOT call model.cuda() — it conflicts with accelerate's device placement.
    # Use backbone/model params for device detection (top-level params like ema_decay_logit may be on CPU).
    if hasattr(model, 'backbone'):
        device = next(model.backbone.parameters()).device
    elif hasattr(model, 'model'):
        device = next(model.model.parameters()).device
    else:
        device = next(model.parameters()).device
    logger.info(f"Model on: {device}")
    model.eval()

    # Sort by text length to minimize padding waste within each batch.
    # Restores original order via sort_idx when saving.
    sort_idx = sorted(range(len(all_texts)), key=lambda j: len(all_texts[j]))
    all_texts = [all_texts[j] for j in sort_idx]
    all_ids = [all_ids[j] for j in sort_idx]

    # Encode in batches with progress bar
    all_dense = []
    all_sparse = []
    # all_steps accumulators
    all_repr_hidden = []     # list of [B, n_gen, H]
    all_sparse_indices = []  # list of [B, n_gen, topk]
    all_sparse_values = []   # list of [B, K, topk]

    num_batches = math.ceil(len(all_texts) / args.batch_size)

    with torch.inference_mode():
        for i in tqdm(range(0, len(all_texts), args.batch_size), total=num_batches, desc="Encoding"):
            batch_texts = all_texts[i:i + args.batch_size]

            if args.model_type in ('trained_diff', 'trained_ar'):
                device = next(model.backbone.parameters()).device
                input_ids, attention_mask = model.tokenize(batch_texts, is_query=args.is_query)
                result = model.encode(input_ids.to(device), attention_mask.to(device),
                                      is_query=args.is_query,
                                      compute_sparse=args.sparse_topk > 0)
                # PromptReps-style content-token filtering + topk for sparse.
                # Run filter+topk on GPU before transferring — for K=16 with
                # V=152k this turns a 156 MB GPU→CPU transfer into a ~1 MB
                # transfer of the [B, K, topk] result, plus avoids 16 separate
                # filter_sparse rebuilds of the same content mask.
                if args.sparse_topk > 0:
                    content_ids = get_content_token_ids(batch_texts, model.tokenizer)

                    def _build_content_mask(B, V, dev, dtype):
                        m = torch.zeros(B, V, device=dev, dtype=dtype)
                        for i in range(B):
                            if content_ids[i]:
                                ids_t = torch.tensor(list(content_ids[i]),
                                                     device=dev, dtype=torch.long)
                                m[i].scatter_(0, ids_t, 1.0)
                        return m

                    if 'sparse_acts_per_pos' in result:
                        # Per-position sparse [B, K, V] on GPU — filter + topk
                        # there, transfer only the small [B, K, topk] result.
                        sp_per = result.pop('sparse_acts_per_pos')  # GPU, bf16
                        result.pop('sparse_acts', None)
                        B, K_sp, V = sp_per.shape
                        mask = _build_content_mask(B, V, sp_per.device, sp_per.dtype)
                        sp_per.mul_(mask.unsqueeze(1))                        # in-place [B, K, V]
                        sp_vals, sp_idxs = sp_per.topk(args.sparse_topk, dim=-1)  # [B, K, topk]
                        sp_vals = (sp_vals.float() * 100).round().int().float()
                        result['sparse_indices'] = sp_idxs.cpu()
                        result['sparse_values'] = sp_vals.cpu()
                    elif 'sparse_acts' in result:
                        # Fallback: max-pooled [B, V] on GPU → [B, 1, topk]
                        sp = result.pop('sparse_acts')  # GPU
                        B, V = sp.shape
                        mask = _build_content_mask(B, V, sp.device, sp.dtype)
                        sp.mul_(mask)
                        sp_vals, sp_idxs = sp.topk(args.sparse_topk, dim=-1)  # [B, topk]
                        sp_vals = (sp_vals.float() * 100).round().int().float()
                        result['sparse_indices'] = sp_idxs.unsqueeze(1).cpu()
                        result['sparse_values'] = sp_vals.unsqueeze(1).cpu()
            elif args.model_type in ('llama', 'qwen', 'qwen25', 'qwen3'):
                result = model.encode(
                    batch_texts,
                    is_query=args.is_query,
                    batch_size=len(batch_texts),
                    encode_type=args.encode_type,
                    sparse_topk=args.sparse_topk,
                )
                # all_steps with num_pooled_tokens>0: model returns K>1 repr_hidden etc. natively.
                # all_steps with num_pooled_tokens==0 (single-representation, K=1): repackage dense/sparse.
                if args.encode_type == 'all_steps' and 'repr_hidden' not in result:
                    # dense [B, H] → repr_hidden [B, 1, H]
                    if 'dense' in result:
                        result['repr_hidden'] = result.pop('dense').unsqueeze(1)
                    elif 'embeddings' in result:
                        result['repr_hidden'] = result.pop('embeddings').unsqueeze(1)
                    # sparse [B, V] → sparse_indices/values [B, 1, topk]
                    if 'sparse' in result:
                        sp = result.pop('sparse').cpu()
                        sp_vals, sp_idxs = sp.topk(args.sparse_topk, dim=-1)
                        sp_vals = (sp_vals * 100).int().float()  # PromptReps quantization step
                        result['sparse_indices'] = sp_idxs.unsqueeze(1)  # [B, 1, topk]
                        result['sparse_values']  = sp_vals.unsqueeze(1)  # [B, 1, topk]
            else:  # llada2 or dream
                result = model.encode(
                    batch_texts,
                    encoding_mode='promptreps',
                    is_query=args.is_query,
                    batch_size=len(batch_texts),
                    show_progress=False,
                    encode_type=args.encode_type,
                )

            if 'dense' in result:
                all_dense.append(result['dense'].cpu().to(torch.bfloat16))
            elif 'embeddings' in result:
                all_dense.append(result['embeddings'].cpu().to(torch.bfloat16))
            if 'sparse' in result:
                # Apply top-k per batch to avoid accumulating full vocab-sized tensors
                batch_sparse = result['sparse'].cpu()
                if args.sparse_topk > 0:
                    vals, idxs = batch_sparse.topk(args.sparse_topk, dim=-1)
                    all_sparse.append((idxs, vals))
                else:
                    all_sparse.append(batch_sparse)
            # all_steps keys
            if 'repr_hidden' in result:
                all_repr_hidden.append(result['repr_hidden'].cpu().to(torch.bfloat16))
            if 'sparse_indices' in result:
                all_sparse_indices.append(result['sparse_indices'].cpu())
            if 'sparse_values' in result:
                all_sparse_values.append(result['sparse_values'].cpu())

    # Save dense embeddings
    if all_dense and args.encode_type in ('all', 'dense'):
        all_dense = torch.cat(all_dense, dim=0)
        logger.info(f"Dense embedding shape: {all_dense.shape}")
        written = save_shard(dense_path, {'ids': all_ids, 'embeddings': all_dense},
                             use_safetensors=True)
        logger.info(f"Saved dense embeddings to {written}")

    # Save sparse embeddings (top-k already applied per batch)
    if all_sparse and args.encode_type in ('all', 'sparse') and args.sparse_topk > 0:
        topk_indices = torch.cat([t[0] for t in all_sparse], dim=0).to(torch.int32)
        topk_values = torch.cat([t[1] for t in all_sparse], dim=0)

        # Auto-trim trailing zero columns
        if args.sparse_save_min > 0:
            orig_k = topk_values.shape[-1]
            any_nonzero = (topk_values != 0).any(dim=0)  # [topk] bool
            last_nz = (any_nonzero.nonzero()[-1].item() + 1) if any_nonzero.any() else 0
            effective_k = min(max(args.sparse_save_min, last_nz), orig_k)
            if effective_k < orig_k:
                topk_indices = topk_indices[:, :effective_k]
                topk_values = topk_values[:, :effective_k]
                logger.info(f"Sparse trimmed: {orig_k} → {effective_k} columns")

        written = save_shard(sparse_path, {
            'ids': all_ids,
            'indices': topk_indices,   # [N, topk] token IDs (int32)
            'values': topk_values,     # [N, topk] scores
        }, use_safetensors=True)
        logger.info(f"Saved sparse embeddings to {written}")

    # Save all_steps data
    if args.encode_type == 'all_steps' and all_repr_hidden:
        payload = {'ids': all_ids}
        rh = torch.cat(all_repr_hidden, dim=0)   # [N, K, H]

        # Trim trailing zero K positions (AR stop-at-quote pads to num_pooled_tokens=20
        # but most docs only use 3-5 tokens). Zero-norm vectors are masked in eval so
        # this is numerically identical.
        per_pos_nonzero = (rh.abs().sum(dim=(0, 2)) > 0)  # [K] bool
        if per_pos_nonzero.any():
            max_used = int(per_pos_nonzero.nonzero()[-1].item()) + 1
        else:
            max_used = 1
        K_orig = rh.shape[1]
        if max_used < K_orig:
            rh = rh[:, :max_used, :].contiguous()
            logger.info(f"Trimmed repr_hidden K: {K_orig} → {max_used} (zero-padded slots dropped)")
        payload['repr_hidden'] = rh
        logger.info(f"repr_hidden shape: {payload['repr_hidden'].shape}")
        if all_sparse_indices:
            si = torch.cat(all_sparse_indices, dim=0).to(torch.int32)  # [N, K, topk]
            sv = torch.cat(all_sparse_values, dim=0)                   # [N, K, topk]
            # Trim sparse K to match repr_hidden
            if si.shape[1] > max_used:
                si = si[:, :max_used, :]
                sv = sv[:, :max_used, :]

            # Auto-trim trailing zero columns (lossless — zeros don't affect scores)
            if args.sparse_save_min > 0:
                orig_k = sv.shape[-1]
                # Find the last column that has any non-zero value across all docs/positions
                any_nonzero = (sv != 0).any(dim=0).any(dim=0)  # [topk] bool
                if any_nonzero.any():
                    last_nz = any_nonzero.nonzero()[-1].item() + 1
                else:
                    last_nz = 0
                effective_k = max(args.sparse_save_min, last_nz)
                effective_k = min(effective_k, orig_k)
                if effective_k < orig_k:
                    si = si[:, :, :effective_k]
                    sv = sv[:, :, :effective_k]
                    logger.info(f"Sparse trimmed: {orig_k} → {effective_k} columns "
                                f"(last non-zero at {last_nz}, min {args.sparse_save_min})")

            # Max-pool sparse across K positions before saving.
            # Eval already max-pools at runtime (see _build_compact_sparse in evaluate_sweep.py),
            # so saving the pooled form is numerically identical.
            if si.shape[1] > 1:
                vocab_sz = model.tokenizer.vocab_size if hasattr(model, 'tokenizer') else None
                si, sv = pool_sparse_across_k(si, sv, vocab_size=vocab_sz)
                logger.info(f"Pooled sparse: [{si.shape[0]}, {si.shape[1]}, {si.shape[2]}]")

            payload['sparse_indices'] = si
            payload['sparse_values'] = sv
        written_path = save_shard(all_steps_path, payload, use_safetensors=True)
        logger.info(f"Saved all_steps data to {written_path}")

    logger.info("Done!")


if __name__ == '__main__':
    main()
