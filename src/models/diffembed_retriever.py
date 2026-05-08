"""DiffEmbed-style retriever — Zhang et al. 2025 (arxiv 2505.15045).

The simplest possible diffusion-LM retriever: tokenize raw text (no prompts,
no K mask tokens), bidirectional forward, mean-pool the last-layer hidden
states over valid (attention_mask=1) tokens, L2-normalize.  Single-vector,
no sparse, no multi-vector.

Architecture differences from your existing PromptReps-style retriever:

  PromptReps          DiffEmbed
  ──────────          ─────────
  prompt template     raw text (no prompt)
  K [MASK] tokens     no mask tokens
  K hidden vectors    1 mean-pooled vector
  sparse from logits  no sparse
  asymmetric K_q/K_p  symmetric (K = 1)

Output schema mirrors the existing pipeline: returns `repr_hidden` of shape
[B, 1, H] so downstream `single_dense` and `multi_dense` modes both reduce
to the same single mean-pooled vector — making it plug-compatible with
encode_promptreps.py / evaluate_sweep.py / monitor.sh without changes.

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

from .backbone_adapters import get_adapter

logger = logging.getLogger(__name__)


class DiffEmbedRetriever(nn.Module):
    """Zero-shot or LoRA-finetuned DiffEmbed-style diffusion-LM retriever.

    Args:
      model_name:    HF model id (e.g. "Dream-org/Dream-v0-Instruct-7B")
      model_type:    one of {"dream","llada1","llada15","llada2"}
      max_length:    truncation length
      normalize:     L2-normalize the pooled vector (recommended)
      lora_rank:     0 = no LoRA (zero-shot); >0 = wrap backbone in PEFT/LoRA
      lora_alpha:    LoRA scaling
      lora_dropout:  LoRA dropout
      attn_implementation: "flash_attention_2" or fallback
      device_map:    HF accelerate device map
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = 'dream',
        max_length: int = 512,
        normalize: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        attn_implementation: str = 'flash_attention_2',
        device_map: Optional[Union[str, dict]] = 'auto',
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        self.max_length = max_length
        self.normalize = normalize
        self.adapter = get_adapter(model_type)

        # Tokenizer (right-pad — we mean-pool, no left-pad mask trickery needed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Backbone with bidirectional attention (diffusion LMs do this natively)
        logger.info(f"Loading {model_type} backbone from {model_name} (DiffEmbed mode)")
        self.backbone = self.adapter.load_backbone(model_name, device_map=device_map)
        try:
            self.backbone.config._attn_implementation = attn_implementation
        except Exception:
            pass
        logger.info(f"DiffEmbed: bidirectional attention via {model_type} adapter, "
                    f"flash_attn={self.adapter.flash_attn}")

        # Optional LoRA wrap (for trained variant)
        if lora_rank > 0:
            from peft import get_peft_model
            lora_cfg = self.adapter.get_lora_config(lora_rank, lora_alpha, lora_dropout)
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            self.backbone.print_trainable_parameters()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Gradient checkpointing — must go through the adapter because some
        # backbones (LLaDA1/15) don't support HF's standard
        # gradient_checkpointing_enable.  Mirrors trainable_diff_retriever.py.
        if gradient_checkpointing:
            self.backbone.enable_input_require_grads()
            self.adapter.enable_gradient_checkpointing(self.backbone)

        # Hidden-state hook on lm_head / ff_out — captures the input to the
        # output projection (== last hidden state) so we can run the backbone
        # with output_hidden_states=False (saves ~1.2 GB activation memory
        # vs keeping all 32 layers' hidden states).
        self._last_hidden: Dict[str, torch.Tensor] = {}
        self._hook_registered = self.adapter.register_hidden_hook(
            self.backbone, self._last_hidden)

        # Hidden size for downstream
        self.hidden_size = getattr(self.backbone.config, 'hidden_size',
                                    getattr(self.backbone.config, 'd_model', None))
        if self.hidden_size is None:
            raise RuntimeError(f"Could not infer hidden_size from {type(self.backbone).__name__}")

        # K=1 markers so monitor.sh / evaluate_sweep / encode pipelines treat
        # us as a single-vector retriever (single_dense == multi_dense reduce
        # to the same vector).
        self.n_gen_tokens = 1
        self.n_gen_q_tokens = 1
        self.n_gen_p_tokens = 1
        self.num_denoise_steps = 1

    # ── Tokenization (no prompts, no mask tokens) ─────────────────────────
    def tokenize(self, texts: Union[str, List[str]], is_query: bool = False, **_):
        """Plain tokenization — DiffEmbed has no prompts and no K mask tokens."""
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            texts,
            padding=True, truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return enc['input_ids'], enc['attention_mask']

    # ── Forward / encode ──────────────────────────────────────────────────
    def _build_4d_mask(self, seq_len: int, attention_mask: torch.Tensor) -> torch.Tensor:
        """4D [B,1,S,S] bidirectional padding mask. Required for Dream
        (Qwen2 architecture defaults to causal) and for LLaDA2 without flash."""
        dtype = next(self.backbone.parameters()).dtype
        min_val = torch.finfo(dtype).min
        B = attention_mask.size(0)
        m = torch.zeros(B, 1, seq_len, seq_len,
                        device=attention_mask.device, dtype=dtype)
        pad = ~attention_mask.bool()
        m = m.masked_fill(pad.unsqueeze(1).unsqueeze(1), min_val)
        m = m.masked_fill(pad.unsqueeze(1).unsqueeze(3), min_val)
        return m

    def _backbone_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Run the backbone and return last_hidden_state [B, L, H].

        Diffusion LMs are bidirectional, but Dream is loaded as Qwen2 (causal
        by default) — pass a 4D mask via the adapter to enforce bidirectional.
        LLaDA1/15 handle bidirectional internally (2D mask). LLaDA2 needs 4D
        unless flash-attn handles masking.
        """
        use_4d = self.adapter.needs_4d_mask()
        if use_4d:
            fwd_mask = self._build_4d_mask(input_ids.size(1), attention_mask)
        else:
            fwd_mask = attention_mask

        # Always pass output_hidden_states=False — three priority paths to
        # extract the last hidden state, all of which work without it:
        #   (1) hidden-state hook (LLaDA1/15/2 — captures input to ff_out/lm_head)
        #   (2) out.last_hidden_state (Dream — AutoModel returns it natively)
        #   (3) out.hidden_states[-1] (rare fallback)
        out = self.backbone(input_ids=input_ids,
                            attention_mask=fwd_mask,
                            output_hidden_states=False,
                            return_dict=True)
        if self._hook_registered and 'h' in self._last_hidden:
            return self._last_hidden.pop('h')
        if getattr(out, 'last_hidden_state', None) is not None:
            return out.last_hidden_state
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            return out.hidden_states[-1]
        # Last resort: rerun with output_hidden_states=True
        out = self.backbone(input_ids=input_ids, attention_mask=fwd_mask,
                            output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1]

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean over valid (attention_mask=1) positions. [B, L, H] → [B, H].

        Reduction is in fp32 to avoid bf16 drift over long sequences (matches
        trainable_diff_retriever._mean_pool's fp32 reduction pattern).

        Memory: instead of casting ``hidden`` to fp32 up-front (a [B, L, H]
        fp32 copy), we keep the masking multiply in bf16 and let
        ``sum(dtype=fp32)`` accumulate in fp32 internally. The mask multiply
        by 0/1 is bit-exact in any float format, so this gives identical
        numerics with ~half the activation memory.
        """
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        masked = hidden * m
        pooled = masked.sum(dim=1, dtype=torch.float32)
        cnt = attention_mask.sum(dim=1, dtype=torch.float32).clamp_min(1.0).unsqueeze(-1)
        return pooled / cnt

    def _encode_tensors(self, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward + mean-pool. Returns pooled fp32 [B, H].

        Uses ``torch.no_grad`` (not ``inference_mode``) so the lazy caches
        some backbones build on first forward (e.g. LLaDA's rotary
        ``pos_cos``/``pos_sin`` buffers) remain usable under autograd in a
        later forward_train call. ``inference_mode`` would mark those cached
        tensors as inference-only and break subsequent training forwards.
        """
        with torch.no_grad():
            hidden = self._backbone_forward(input_ids, attention_mask)
            pooled = self._mean_pool(hidden, attention_mask)
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

        Signature mirrors DreamRetriever.encode / LLaDARetriever.encode so
        the existing ``else:`` branch in scripts/encode_promptreps.py works
        without a special-case dispatch.
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
            # [B, H] → [B, 1, H]  (K-axis = 1 for plug-compat with multi_dense)
            all_repr.append(pooled.unsqueeze(1).to(torch.bfloat16).cpu())

        repr_hidden = torch.cat(all_repr, dim=0) if all_repr else \
                      torch.empty(0, 1, self.hidden_size, dtype=torch.bfloat16)
        return {'repr_hidden': repr_hidden}

    # ── Training-mode forward (used by train_diffembed.py) ────────────────
    def forward_train(self,
                      query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                      passage_input_ids: torch.Tensor, passage_attention_mask: torch.Tensor,
                      labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE on mean-pooled (q, p) vectors.

        Args:
          query_input_ids/mask:   [B_q, L_q]
          passage_input_ids/mask: [B_p, L_p]   (B_p = B_q × (1 + n_neg))
          labels:                 [B_q]   index of positive p per query

        Returns:
          {"loss": scalar, "loss_dense": scalar}
        """
        q_h = self._backbone_forward(query_input_ids, query_attention_mask)
        p_h = self._backbone_forward(passage_input_ids, passage_attention_mask)
        q = self._mean_pool(q_h, query_attention_mask)
        p = self._mean_pool(p_h, passage_attention_mask)
        if self.normalize:
            q = F.normalize(q, p=2, dim=-1)
            p = F.normalize(p, p=2, dim=-1)
        # Scaled dot-product, InfoNCE
        scores = q @ p.T / 0.02              # τ=0.02, matches PromptReps default
        loss = F.cross_entropy(scores, labels)
        return {'loss': loss, 'loss_dense': loss.detach()}

    # ── HF Trainer / DeepSpeed integration ────────────────────────────────
    @property
    def config(self):
        """HF Trainer + DeepSpeed expect ``model.config`` to exist."""
        return self.backbone.config

    def gradient_checkpointing_enable(self, **kwargs):
        """Route through the adapter so LLaDA1/15's custom block-wrapping path
        is used (their LLaDAModelLM doesn't support HF's standard call).

        HF Trainer calls this when ``TrainingArguments(gradient_checkpointing=True)``.
        Without this override, the call lands on the bare backbone and raises
        ``ValueError: LLaDAModelLM does not support gradient checkpointing.``
        """
        if self.adapter:
            self.adapter.enable_gradient_checkpointing(self.backbone, **kwargs)
        else:
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """No-op fallback — adapter doesn't expose disable.  HF Trainer never
        calls this in normal training, but defining it keeps duck-typing happy."""
        if hasattr(self.backbone, 'gradient_checkpointing_disable'):
            try:
                self.backbone.gradient_checkpointing_disable()
            except (AttributeError, ValueError):
                pass

    # ── Save / load (LoRA-only, similar to TrainableDiffusionRetriever) ───
    def save(self, output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # LoRA adapters (or full model if lora_rank=0)
        self.backbone.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        cfg = {
            'pooling': 'mean',
            'model_type': self.model_type,
            'max_length': self.max_length,
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
        with open(output_dir / 'diffembed_config.json', 'w') as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"Saved DiffEmbed retriever to {output_dir}")

    @classmethod
    def load(cls, model_dir: Union[str, Path], **fallback_kwargs) -> 'DiffEmbedRetriever':
        """Load a saved DiffEmbed checkpoint (with LoRA merged or zero-shot).

        Build with ``lora_rank=0`` (bare backbone), then attach the saved
        LoRA via ``PeftModel.from_pretrained`` and ``merge_and_unload``.
        Building with ``lora_rank>0`` first would wrap the backbone with a
        random LoRA, and the saved adapter would then double-wrap on top —
        ``merge_and_unload`` would merge only the outer adapter and the
        inner random LoRA would silently perturb the forward.
        """
        import json
        model_dir = Path(model_dir)
        cfg_path = model_dir / 'diffembed_config.json'
        # Caller-supplied kwargs WIN over saved config — important so
        # encode_promptreps.py (which passes --max_length 512) gets the
        # standard inference recipe rather than the training-time 156.
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            model_type = fallback_kwargs.get('model_type', cfg['model_type'])
            max_length = fallback_kwargs.get('max_length', cfg.get('max_length', 512))
            normalize  = fallback_kwargs.get('normalize',  cfg.get('normalize', True))
            saved_lora_rank = cfg.get('lora_rank', 0)
            saved_lora_alpha = cfg.get('lora_alpha', 64)
        else:
            model_type = fallback_kwargs.get('model_type', 'dream')
            max_length = fallback_kwargs.get('max_length', 512)
            normalize = fallback_kwargs.get('normalize', True)
            saved_lora_rank = fallback_kwargs.get('lora_rank', 0)
            saved_lora_alpha = fallback_kwargs.get('lora_alpha', 64)

        adapter = get_adapter(model_type)
        # If LoRA was saved, the model_dir holds only adapter weights — base
        # comes from the hub.  If lora_rank=0, model_dir holds the full model.
        source = adapter.hub_model_name if saved_lora_rank > 0 else str(model_dir)
        retriever = cls(
            model_name=source,
            model_type=model_type,
            max_length=max_length,
            normalize=normalize,
            lora_rank=0,                     # bare — we attach saved LoRA below
            attn_implementation=fallback_kwargs.get('attn_implementation', 'flash_attention_2'),
            device_map=fallback_kwargs.get('device_map', 'auto'),
        )

        if saved_lora_rank > 0:
            from peft import PeftModel
            retriever.backbone = PeftModel.from_pretrained(retriever.backbone, str(model_dir))
            retriever.backbone = retriever.backbone.merge_and_unload()
            # Hook was registered against the pre-merge backbone — re-register
            # against the merged backbone so it captures hidden states correctly.
            retriever._last_hidden = {}
            retriever._hook_registered = retriever.adapter.register_hidden_hook(
                retriever.backbone, retriever._last_hidden)
            logger.info(f"Loaded + merged LoRA adapters from {model_dir}")

        # Restore saved hyperparams (used by save() roundtrip + introspection)
        retriever.lora_rank = saved_lora_rank
        retriever.lora_alpha = saved_lora_alpha

        retriever.eval()
        return retriever
