"""RepLLaMA-style retriever (Ma et al. 2024, arxiv 2310.08319).

Causal LLaMA backbone with the EOS token appended to every input text. The
hidden state at the EOS position is the embedding — RepLLaMA is a
single-representation (K=1) retriever by design (NOT mean-pool, NOT K-vec),
training the model to encode context into that single position via InfoNCE.

Architectural differences from DiffEmbed (the diffusion-LM analog):

  DiffEmbed                RepLLaMA
  ─────────                ────────
  diffusion backbone       causal AR backbone (LLaMA / Mistral / Qwen)
  bidirectional attn       causal attn (NO bidirectional swap)
  mean-pool over tokens    last-token (EOS) pool
  no special tokens        EOS token appended to every input
  K=1 single vector        K=1 single vector

Output schema mirrors the existing pipeline: returns ``repr_hidden`` of shape
[B, 1, H] so downstream ``single_dense`` and ``multi_dense`` modes both reduce
to the same vector — plug-compatible with encode.py /
evaluate_sweep.py / monitor.sh.

Both zero-shot and trained (LoRA) variants are supported via this one class.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


# Standard LoRA target modules for LLaMA-family models (Llama-2, Llama-3, Mistral)
LLAMA_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class RepLLaMARetriever(nn.Module):
    """Zero-shot or LoRA-finetuned RepLLaMA-style dense retriever.

    Args:
      model_name:    HF model id (default: meta-llama/Meta-Llama-3-8B-Instruct)
      max_length:    truncation length (text + EOS must fit)
      normalize:     L2-normalize the pooled vector (recommended)
      lora_rank:     0 = no LoRA (zero-shot); >0 = wrap backbone in PEFT/LoRA
      lora_alpha:    LoRA scaling
      lora_dropout:  LoRA dropout
      attn_implementation: "flash_attention_2" or fallback
      device_map:    HF accelerate device map
      gradient_checkpointing: enable HF gradient checkpointing on the backbone
    """

    def __init__(
        self,
        model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct',
        max_length: int = 512,
        normalize: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        attn_implementation: str = 'flash_attention_2',
        device_map: Optional[Union[str, dict]] = 'auto',
        gradient_checkpointing: bool = False,
        query_max_len: Optional[int] = None,
        passage_max_len: Optional[int] = None,
        query_prefix: str = '',
        passage_prefix: str = '',
        lora_targets: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        # Side-marker prefixes (RepLLaMA paper, Ma 2024).  Empty → no prefix
        # added.  Common values: 'query: ' and 'passage: '.
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        # Per-side truncation lengths.  Default to ``max_length`` for both
        # sides — matches the standard inference recipe used by all other
        # retrievers in this codebase (PromptReps/DiffRetriever run at 512/512).
        # The collator overrides these per call during training, so this
        # default doesn't affect training.
        self.query_max_len = query_max_len if query_max_len is not None else max_length
        self.passage_max_len = passage_max_len if passage_max_len is not None else max_length

        # Tokenizer.  LEFT-PAD — for last-token retrievers this is critical:
        # the appended EOS always lands at position max_len-1 (the very last
        # token of every sample), giving the model a consistent RoPE position
        # for the pool anchor across all sequence lengths and across train
        # and inference.  Right-padding would put the EOS at variable
        # positions per sample, adding noise that hurts OOD generalization.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._eos_id = self.tokenizer.eos_token_id

        # Backbone.  AutoModel returns last_hidden_state natively (no lm_head
        # matmul wasted) — same efficiency story as Dream in DiffEmbed.
        logger.info(f"Loading RepLLaMA backbone from {model_name}")
        common_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
        if device_map is not None:
            common_kw['device_map'] = device_map
        try:
            self.backbone = AutoModel.from_pretrained(
                model_name, attn_implementation=attn_implementation, **common_kw)
            self.flash_attn = True
            logger.info(f"RepLLaMA: {attn_implementation} enabled")
        except (ValueError, ImportError):
            self.backbone = AutoModel.from_pretrained(model_name, **common_kw)
            self.flash_attn = False
            logger.info("RepLLaMA: flash attention not available, using eager")

        # LoRA wrap for trained variant
        # Default targets include MLP layers — matches DiffEmbed and DiffRetriever
        # so the paper's "is multi-vector / mean-pool / last-token-pool the
        # right inductive bias?" comparison is at fixed LoRA-capacity.
        self.lora_targets = lora_targets if lora_targets is not None \
                            else LLAMA_LORA_TARGETS
        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=self.lora_targets,
                task_type=TaskType.FEATURE_EXTRACTION,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            self.backbone.print_trainable_parameters()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Gradient checkpointing — Llama supports HF's standard call directly
        # (no custom adapter needed, unlike LLaDA1 in DiffEmbed).
        if gradient_checkpointing:
            self.backbone.enable_input_require_grads()
            self.backbone.gradient_checkpointing_enable()

        # Hidden size for downstream
        self.hidden_size = self.backbone.config.hidden_size

        # K=1 markers so monitor.sh / evaluate_sweep / encode pipelines treat
        # this as a single-vector retriever (single_dense == multi_dense reduce
        # to the same vector — RepLLaMA returns one EOS token embedding).
        self.n_gen_tokens = 1
        self.n_gen_q_tokens = 1
        self.n_gen_p_tokens = 1
        self.num_denoise_steps = 1

    # ── HF Trainer / DeepSpeed integration ────────────────────────────────
    @property
    def config(self):
        """HF Trainer + DeepSpeed expect ``model.config`` to exist."""
        return self.backbone.config

    def gradient_checkpointing_enable(self, **kwargs):
        """Llama supports the standard call — pass through to backbone."""
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, 'gradient_checkpointing_disable'):
            try:
                self.backbone.gradient_checkpointing_disable()
            except (AttributeError, ValueError):
                pass

    # ── Tokenization (append EOS) ─────────────────────────────────────────
    def tokenize(self, texts: Union[str, List[str]], is_query: bool = False, **_):
        """Tokenize and append EOS token id.  RepLLaMA's training objective
        teaches the model to encode the document into the EOS position.

        Uses per-side truncation length:
          * is_query=True  → self.query_max_len   (default 32)
          * is_query=False → self.passage_max_len (default 156)
        Critical for inference correctness: the model was trained with the EOS
        at distinct position ranges per side, so encoding queries with the
        passage budget shifts the EOS position out of distribution and
        degrades retrieval — visible especially on BEIR datasets with long
        queries (ArguAna, FiQA).
        """
        if isinstance(texts, str):
            texts = [texts]
        max_len_side = self.query_max_len if is_query else self.passage_max_len
        # Side-marker prefix (RepLLaMA paper recipe): "query: " or "passage: "
        prefix = self.query_prefix if is_query else self.passage_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        # Tokenize with room for the trailing EOS (max_length-1 leaves space).
        # add_special_tokens=True keeps BOS at the start (Llama convention).
        enc = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_len_side - 1,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        eos_id = self._eos_id
        pad_id = self.tokenizer.pad_token_id
        ids_list = [list(ids) + [eos_id] for ids in enc['input_ids']]
        max_len = max(len(ids) for ids in ids_list)
        # LEFT-PAD: pads at the START of each row, real tokens (ending in EOS)
        # at the END.  Means EOS is always at position max_len-1.
        input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        for i, ids in enumerate(ids_list):
            n = len(ids)
            input_ids[i, max_len - n:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - n:] = 1
        return input_ids, attention_mask

    # ── Forward / encode ──────────────────────────────────────────────────
    def _backbone_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Causal forward.  Returns last_hidden_state [B, L, H] (bf16)."""
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                            return_dict=True)
        if getattr(out, 'last_hidden_state', None) is not None:
            return out.last_hidden_state
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            return out.hidden_states[-1]
        # Last resort
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1]

    def _last_token_pool(self, hidden: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
        """Index the hidden state at the EOS position.  With LEFT-padding
        the EOS is always at the last index of the sequence dimension —
        a single fixed position across the batch.

        For backward compatibility with old RIGHT-padded checkpoints,
        falls back to ``attention_mask.sum() - 1`` if padding is on the right.

        Returns fp32 [B, H] (cast for L2-normalize stability).
        """
        if self.tokenizer.padding_side == 'left':
            return hidden[:, -1, :].float()
        # Right-pad fallback (old checkpoints)
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, seq_lens].float()

    def _encode_tensors(self, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward + last-token pool. Returns fp32 [B, H]."""
        with torch.no_grad():
            hidden = self._backbone_forward(input_ids, attention_mask)
            pooled = self._last_token_pool(hidden, attention_mask)
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def encode(self, texts, *,
               batch_size: int = 32,
               is_query: bool = False,
               encode_type: str = 'all_steps',
               encoding_mode: str = 'promptreps',
               show_progress: bool = False,
               **_) -> Dict[str, torch.Tensor]:
        """Pipeline-facing encode: takes a list of strings, returns
        ``{'repr_hidden': [B, 1, H] bf16}``.

        Signature mirrors DiffEmbedRetriever.encode for plug-compatibility
        with scripts/encode.py's existing dispatch.
        """
        if isinstance(texts, str):
            texts = [texts]
        device = next(self.backbone.parameters()).device

        all_repr = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            input_ids, attention_mask = self.tokenize(chunk, is_query=is_query)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pooled = self._encode_tensors(input_ids, attention_mask)
            # [B, H] → [B, 1, H]  (K-axis = 1 for plug-compat)
            all_repr.append(pooled.unsqueeze(1).to(torch.bfloat16).cpu())

        repr_hidden = torch.cat(all_repr, dim=0) if all_repr else \
                      torch.empty(0, 1, self.hidden_size, dtype=torch.bfloat16)
        return {'repr_hidden': repr_hidden}

    # ── Training-mode forward (used by scripts/train_repllama.py) ─────────
    def forward_train(self,
                      query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                      passage_input_ids: torch.Tensor, passage_attention_mask: torch.Tensor,
                      labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE on last-token-pooled (q, p) vectors.

        Identical to DiffEmbedRetriever.forward_train except the pool function
        is last-token (EOS) instead of mean-pool, and attention is causal.

        Args:
          query_input_ids/mask:   [B_q, L_q]
          passage_input_ids/mask: [B_p, L_p]   (B_p = B_q × (1 + n_neg))
          labels:                 [B_q]   index of positive p per query

        Returns:
          {"loss": scalar, "loss_dense": scalar}
        """
        q_h = self._backbone_forward(query_input_ids, query_attention_mask)
        p_h = self._backbone_forward(passage_input_ids, passage_attention_mask)
        q = self._last_token_pool(q_h, query_attention_mask)
        p = self._last_token_pool(p_h, passage_attention_mask)
        if self.normalize:
            q = F.normalize(q, p=2, dim=-1)
            p = F.normalize(p, p=2, dim=-1)
        # τ=0.02 — matches DiffEmbed and PromptReps default
        scores = q @ p.T / 0.02
        loss = F.cross_entropy(scores, labels)
        return {'loss': loss, 'loss_dense': loss.detach()}

    # ── Save / load ───────────────────────────────────────────────────────
    def save(self, output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # LoRA adapters (or full model if lora_rank=0)
        self.backbone.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        cfg = {
            'pooling': 'last_token',
            'model_name': self.model_name,
            'max_length': self.max_length,
            'query_max_len': self.query_max_len,
            'passage_max_len': self.passage_max_len,
            'query_prefix': self.query_prefix,
            'passage_prefix': self.passage_prefix,
            'lora_targets': self.lora_targets,
            'normalize': self.normalize,
            'hidden_size': self.hidden_size,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'n_gen_tokens': 1,
            'n_gen_q_tokens': 1,
            'n_gen_p_tokens': 1,
            'num_denoise_steps': 1,
        }
        import json
        with open(output_dir / 'repllama_config.json', 'w') as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"Saved RepLLaMA retriever to {output_dir}")

    @classmethod
    def load(cls, model_dir: Union[str, Path], **fallback_kwargs) -> 'RepLLaMARetriever':
        """Load a saved RepLLaMA checkpoint (with LoRA merged or zero-shot).

        Build with ``lora_rank=0`` (bare backbone), then attach the saved LoRA
        via ``PeftModel.from_pretrained`` and ``merge_and_unload``.  Mirrors
        DiffEmbedRetriever.load to avoid the random-LoRA double-wrap bug.
        """
        import json
        model_dir = Path(model_dir)
        cfg_path = model_dir / 'repllama_config.json'
        # Caller-supplied kwargs WIN over saved config — important so
        # encode.py (which passes --max_length 512) gets the
        # standard inference recipe rather than the training-time 156.
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            model_name = fallback_kwargs.get(
                'model_name', cfg.get('model_name', 'meta-llama/Meta-Llama-3-8B-Instruct'))
            max_length = fallback_kwargs.get('max_length', cfg.get('max_length', 512))
            normalize = fallback_kwargs.get('normalize', cfg.get('normalize', True))
            saved_lora_rank = cfg.get('lora_rank', 0)
            saved_lora_alpha = cfg.get('lora_alpha', 64)
            saved_query_prefix = cfg.get('query_prefix', '')
            saved_passage_prefix = cfg.get('passage_prefix', '')
        else:
            model_name = fallback_kwargs.get('model_name',
                                             'meta-llama/Meta-Llama-3-8B-Instruct')
            max_length = fallback_kwargs.get('max_length', 512)
            normalize = fallback_kwargs.get('normalize', True)
            saved_lora_rank = fallback_kwargs.get('lora_rank', 0)
            saved_lora_alpha = fallback_kwargs.get('lora_alpha', 64)
            saved_query_prefix = ''
            saved_passage_prefix = ''

        # If LoRA was saved, the model_dir holds only adapter weights — base
        # comes from the hub.  If lora_rank=0, model_dir holds the full model.
        source = model_name if saved_lora_rank > 0 else str(model_dir)
        retriever = cls(
            model_name=source,
            max_length=max_length,
            normalize=normalize,
            lora_rank=0,                  # bare — attach LoRA below if needed
            attn_implementation=fallback_kwargs.get('attn_implementation', 'flash_attention_2'),
            device_map=fallback_kwargs.get('device_map', 'auto'),
            query_prefix=saved_query_prefix,
            passage_prefix=saved_passage_prefix,
        )

        if saved_lora_rank > 0:
            from peft import PeftModel
            retriever.backbone = PeftModel.from_pretrained(retriever.backbone, str(model_dir))
            retriever.backbone = retriever.backbone.merge_and_unload()
            logger.info(f"Loaded + merged LoRA adapters from {model_dir}")
            if saved_query_prefix or saved_passage_prefix:
                logger.info(f"Using prefixes: query={saved_query_prefix!r} "
                            f"passage={saved_passage_prefix!r}")

        retriever.lora_rank = saved_lora_rank
        retriever.lora_alpha = saved_lora_alpha

        retriever.eval()
        return retriever
