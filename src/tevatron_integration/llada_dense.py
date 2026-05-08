"""
LLaDA 2 Dense Model — Tevatron-compatible

Wraps LLaDA 2's block diffusion retriever to conform to Tevatron's
EncoderModel interface for training, encoding, and evaluation pipelines.

No projection layer — embeddings are raw hidden states from the backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM
import logging
import json
import os

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / 'tevatron' / 'src'))
from tevatron.retriever.modeling.encoder import EncoderModel

from .arguments import LLaDA2ModelArguments, LLaDA2TrainingArguments
from ..models.block_schedule import BlockSchedule

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 156895  # LLaDA 2's <|mask|> token id


class LLaDA2DenseModel(EncoderModel):
    """
    Tevatron-compatible dense retrieval model using LLaDA 2 block diffusion.

    No projection — uses raw hidden states as embeddings.
    """

    def __init__(
        self,
        encoder,
        pooling: str = 'mean',
        normalize: bool = True,
        temperature: float = 0.02,
        encoding_mode: str = 'clean',
        block_schedule: Optional[BlockSchedule] = None,
        block_aggregation: str = 'ema',
        mask_token_id: int = MASK_TOKEN_ID,
        num_repr_tokens: int = 1,
        num_denoise_steps: int = 1,
        query_prefix: str = "",
        query_suffix: str = "",
        passage_prefix: str = "",
        passage_suffix: str = "",
    ):
        super().__init__(encoder, pooling, normalize, temperature)

        self.pooling = pooling
        self.normalize = normalize
        self.encoding_mode = encoding_mode
        self.block_schedule = block_schedule or BlockSchedule()
        self.block_aggregation = block_aggregation
        self.mask_token_id = mask_token_id
        self.num_repr_tokens = num_repr_tokens
        self.num_denoise_steps = num_denoise_steps
        self.query_prefix = query_prefix
        self.query_suffix = query_suffix
        self.passage_prefix = passage_prefix
        self.passage_suffix = passage_suffix

        # Determine hidden size (= embedding dim)
        self.hidden_size = getattr(
            encoder.config, 'hidden_size',
            getattr(encoder.config, 'd_model', 2048)
        )

        # Attention pooling (optional)
        if pooling == "attention":
            self.attn_pool = nn.Sequential(nn.Linear(self.hidden_size, 1))

        # Block aggregation layers
        if block_aggregation == "ema":
            self.ema_decay_logit = nn.Parameter(torch.tensor(1.0))
        elif block_aggregation == "attention":
            self.block_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_size, num_heads=8, batch_first=True
            )
            self.block_query = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pooling(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Pool token representations into sequence representation."""
        if self.pooling in ['cls', 'first']:
            return hidden[:, 0]

        elif self.pooling in ['mean', 'avg']:
            m = mask.unsqueeze(-1).float()
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

        elif self.pooling == 'weighted_mean':
            seq_len = hidden.size(1)
            w = torch.arange(1, seq_len + 1, device=hidden.device, dtype=torch.float)
            w = w.unsqueeze(0).unsqueeze(-1) * mask.unsqueeze(-1).float()
            return (hidden * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-9)

        elif self.pooling in ['last', 'eos']:
            seq_lengths = mask.sum(dim=1) - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, seq_lengths]

        elif self.pooling == 'attention':
            attn_w = self.attn_pool(hidden).squeeze(-1)
            attn_w = attn_w.masked_fill(~mask.bool(), float('-inf'))
            attn_w = torch.softmax(attn_w, dim=-1).unsqueeze(-1)
            return (hidden * attn_w).sum(dim=1)

        raise ValueError(f"Unknown pooling: {self.pooling}")

    # ------------------------------------------------------------------
    # Hidden state extraction
    # ------------------------------------------------------------------

    def _get_hidden_states(
        self, input_ids: Tensor, attention_mask: Tensor,
        attention_mask_4d: Optional[Tensor] = None,
    ) -> Tensor:
        """Extract last-layer hidden states from LLaDA 2."""
        kwargs = dict(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        if attention_mask_4d is not None:
            kwargs["attention_mask"] = attention_mask_4d
        else:
            kwargs["attention_mask"] = attention_mask

        outputs = self.encoder(**kwargs)

        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            return outputs.hidden_states[-1].float()
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state.float()
        raise RuntimeError("LLaDA 2 did not return hidden states")

    # ------------------------------------------------------------------
    # Block-causal mask
    # ------------------------------------------------------------------

    def _build_block_causal_mask(
        self, seq_len: int, attention_mask: Tensor,
    ) -> Tensor:
        """Build [B, 1, S, S] block-causal attention mask."""
        device = attention_mask.device
        block_len = self.block_schedule.block_length

        positions = torch.arange(seq_len, device=device)
        block_ids = positions // block_len

        causal_mask = block_ids.unsqueeze(0) <= block_ids.unsqueeze(1)
        mask_2d = torch.where(causal_mask, 0.0, float('-inf'))

        mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).expand(
            attention_mask.size(0), 1, seq_len, seq_len
        ).clone()

        pad_mask_key = ~attention_mask.bool().unsqueeze(1).unsqueeze(1)
        mask_4d = mask_4d.masked_fill(pad_mask_key, float('-inf'))

        return mask_4d

    # ------------------------------------------------------------------
    # Block embedding aggregation
    # ------------------------------------------------------------------

    def _aggregate_block_embeddings(self, block_embeddings: List[Tensor]) -> Tensor:
        """Aggregate progressive block embeddings into final embedding."""
        if len(block_embeddings) == 1:
            return block_embeddings[0]

        if self.block_aggregation == "last":
            return block_embeddings[-1]

        stacked = torch.stack(block_embeddings, dim=1)

        if self.block_aggregation == "mean":
            return stacked.mean(dim=1)

        elif self.block_aggregation == "weighted_mean":
            K = stacked.size(1)
            weights = torch.arange(1, K + 1, device=stacked.device, dtype=torch.float)
            weights = weights / weights.sum()
            return (stacked * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

        elif self.block_aggregation == "ema":
            decay = torch.sigmoid(self.ema_decay_logit)
            K = stacked.size(1)
            ema_weights = torch.zeros(K, device=stacked.device)
            for k in range(K):
                ema_weights[k] = (1 - decay) * decay ** (K - 1 - k)
            ema_weights = ema_weights / ema_weights.sum()
            return (stacked * ema_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

        elif self.block_aggregation == "attention":
            query = self.block_query.expand(stacked.size(0), -1, -1)
            out, _ = self.block_attn(query, stacked, stacked)
            return out.squeeze(1)

        raise ValueError(f"Unknown block aggregation: {self.block_aggregation}")

    # ------------------------------------------------------------------
    # Encoding modes
    # ------------------------------------------------------------------

    def _encode_clean(self, text_input: Dict[str, Tensor]) -> Tensor:
        """Clean encoding — LLaDA 2 as bidirectional encoder."""
        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']
        hidden = self._get_hidden_states(input_ids, attention_mask)
        return self._pooling(hidden, attention_mask)

    def _encode_block_interactive(self, text_input: Dict[str, Tensor]) -> Tensor:
        """Block-interactive encoding — progressive embeddings via block-causal mask."""
        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']
        seq_len = input_ids.size(1)

        block_causal_mask = self._build_block_causal_mask(seq_len, attention_mask)
        hidden = self._get_hidden_states(
            input_ids, attention_mask, attention_mask_4d=block_causal_mask
        )

        boundaries = self.block_schedule.get_block_boundaries(seq_len)
        block_embeddings = []

        for k, (_, end) in enumerate(boundaries):
            partial_emb = hidden[:, :end, :]
            partial_mask = attention_mask[:, :end]
            pooled = self._pooling(partial_emb, partial_mask)
            block_embeddings.append(pooled)

        return self._aggregate_block_embeddings(block_embeddings)

    @torch.no_grad()
    def _encode_block_denoising(self, text_input: Dict[str, Tensor]) -> Tensor:
        """Block-denoising encoding — actual denoising from [MASK] (inference only)."""
        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']
        device = input_ids.device
        batch_size = input_ids.size(0)
        prompt_len = input_ids.size(1)
        block_len = self.block_schedule.block_length
        num_steps = self.block_schedule.num_steps_per_block

        response_ids = torch.full(
            (batch_size, block_len), self.mask_token_id,
            dtype=input_ids.dtype, device=device,
        )
        response_mask = torch.ones(
            (batch_size, block_len), dtype=attention_mask.dtype, device=device,
        )

        full_ids = torch.cat([input_ids, response_ids], dim=1)
        full_mask = torch.cat([attention_mask, response_mask], dim=1)

        block_start = prompt_len
        block_end = prompt_len + block_len
        block_positions = list(range(block_start, block_end))
        masked_positions = set(block_positions)
        tokens_per_step = max(1, block_len // num_steps)

        for step in range(num_steps):
            if not masked_positions:
                break

            outputs = self.encoder(
                input_ids=full_ids,
                attention_mask=full_mask,
                return_dict=True,
            )
            logits = outputs.logits

            masked_pos_list = sorted(masked_positions)
            pos_tensor = torch.tensor(masked_pos_list, device=device)
            pos_logits = logits[:, pos_tensor, :]
            pos_probs = F.softmax(pos_logits, dim=-1)

            confidences, predicted_tokens = pos_probs.max(dim=-1)

            num_to_reveal = min(tokens_per_step, len(masked_pos_list))
            if step == num_steps - 1:
                num_to_reveal = len(masked_pos_list)

            avg_confidence = confidences.mean(dim=0)
            _, topk_indices = avg_confidence.topk(num_to_reveal)

            for idx in topk_indices:
                pos = masked_pos_list[idx.item()]
                full_ids[:, pos] = predicted_tokens[:, idx]
                masked_positions.discard(pos)

            if self.block_schedule.enable_t2t and step < num_steps - 1:
                revealed = [p for p in block_positions if p not in masked_positions]
                if revealed:
                    rev_tensor = torch.tensor(revealed, device=device)
                    rev_logits = logits[:, rev_tensor, :]
                    full_ids[:, rev_tensor] = rev_logits.argmax(dim=-1)

        # Final embedding from full denoised sequence
        hidden = self._get_hidden_states(full_ids, full_mask)
        return self._pooling(hidden, full_mask)

    # ------------------------------------------------------------------
    # PromptReps encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_promptreps(self, text_input: Dict[str, Tensor]) -> Tensor:
        """PromptReps encoding — prefix + text + suffix + [MASK] → denoise.

        Input should already have [MASK] tokens appended (by dataset/collator).
        The last num_repr_tokens positions are [MASK].

        Single-step: forward pass, hidden states at [MASK] = dense embedding.
        Multi-step: iteratively denoise, mean-pool hidden states across steps.
        """
        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']
        device = input_ids.device
        n_repr = self.num_repr_tokens
        num_steps = self.num_denoise_steps
        seq_len = input_ids.size(1)

        # [MASK] positions are the last n_repr tokens
        repr_positions = torch.arange(seq_len - n_repr, seq_len, device=device)

        all_hidden = []
        curr_ids = input_ids.clone()

        for step in range(num_steps):
            hidden = self._get_hidden_states(curr_ids, attention_mask)
            repr_hidden = hidden[:, repr_positions, :]

            if n_repr == 1:
                step_emb = repr_hidden.squeeze(1)
            else:
                step_emb = repr_hidden.mean(dim=1)
            all_hidden.append(step_emb)

            # Multi-step: denoise [MASK] → reveal → re-encode
            if step < num_steps - 1:
                outputs = self.encoder(
                    input_ids=curr_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                pred_tokens = outputs.logits[:, repr_positions, :].argmax(dim=-1)
                curr_ids[:, repr_positions] = pred_tokens

        if len(all_hidden) == 1:
            return all_hidden[0]
        return torch.stack(all_hidden, dim=1).mean(dim=1)

    # ------------------------------------------------------------------
    # Core encode dispatch
    # ------------------------------------------------------------------

    def _encode_text(self, text_input: Dict[str, Tensor]) -> Tensor:
        """Core encoding with mode selection."""
        if self.encoding_mode == "block_interactive":
            emb = self._encode_block_interactive(text_input)
        elif self.encoding_mode == "promptreps":
            emb = self._encode_promptreps(text_input)
        elif self.encoding_mode == "block_denoising":
            emb = self._encode_block_denoising(text_input)
        else:  # clean
            emb = self._encode_clean(text_input)

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb

    def encode_query(self, qry: Dict[str, Tensor]) -> Tensor:
        return self._encode_text(qry)

    def encode_passage(self, psg: Dict[str, Tensor]) -> Tensor:
        return self._encode_text(psg)

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    # ------------------------------------------------------------------
    # Build / Load / Save
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, model_args: LLaDA2ModelArguments, train_args: LLaDA2TrainingArguments, **hf_kwargs):
        """Build model from arguments."""
        logger.info(f"Loading LLaDA 2 from {model_args.model_name_or_path}")

        hf_kwargs.setdefault('trust_remote_code', True)
        hf_kwargs.setdefault('torch_dtype', torch.bfloat16)
        hf_kwargs.setdefault('device_map', 'auto')

        base_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **hf_kwargs
        )

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        if train_args.gradient_checkpointing:
            base_model.enable_input_require_grads()

        schedule = BlockSchedule(
            block_length=model_args.block_length,
            num_steps_per_block=model_args.num_steps_per_block,
            enable_t2t=model_args.enable_t2t,
        )

        model = cls(
            encoder=base_model,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            temperature=train_args.temperature,
            encoding_mode=model_args.encoding_mode,
            block_schedule=schedule,
            block_aggregation=model_args.block_aggregation,
            mask_token_id=model_args.mask_token_id,
            num_repr_tokens=model_args.num_repr_tokens,
            num_denoise_steps=model_args.num_denoise_steps,
            query_prefix=model_args.query_prefix,
            query_suffix=model_args.query_suffix,
            passage_prefix=model_args.passage_prefix,
            passage_suffix=model_args.passage_suffix,
        )

        if model_args.freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False
            logger.info("LLaDA 2 backbone frozen")

        return model

    @classmethod
    def load(cls, model_name_or_path: str, **hf_kwargs):
        """Load pre-trained model for inference."""
        hf_kwargs.setdefault('trust_remote_code', True)
        hf_kwargs.setdefault('torch_dtype', torch.bfloat16)
        hf_kwargs.setdefault('device_map', 'auto')

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **hf_kwargs
        )

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        config_path = f"{model_name_or_path}/llada2_retrieval_config.json"
        model_kwargs = {}
        try:
            with open(config_path) as f:
                saved_config = json.load(f)
            schedule_info = saved_config.pop('block_schedule', {})
            model_kwargs = {
                'encoding_mode': saved_config.get('encoding_mode', 'clean'),
                'block_schedule': BlockSchedule(**schedule_info),
                'block_aggregation': saved_config.get('block_aggregation', 'ema'),
                'pooling': saved_config.get('pooling', 'mean'),
                'normalize': saved_config.get('normalize', True),
                'mask_token_id': saved_config.get('mask_token_id', MASK_TOKEN_ID),
                'num_repr_tokens': saved_config.get('num_repr_tokens', 1),
                'num_denoise_steps': saved_config.get('num_denoise_steps', 1),
                'query_prefix': saved_config.get('query_prefix', ''),
                'query_suffix': saved_config.get('query_suffix', ''),
                'passage_prefix': saved_config.get('passage_prefix', ''),
                'passage_suffix': saved_config.get('passage_suffix', ''),
            }
        except FileNotFoundError:
            logger.warning(f"No retrieval config at {config_path}, using defaults")

        model = cls(encoder=base_model, **model_kwargs)

        head_path = f"{model_name_or_path}/retrieval_head.pt"
        try:
            head_state = torch.load(head_path, map_location='cpu', weights_only=True)
            if 'attn_pool' in head_state and hasattr(model, 'attn_pool'):
                model.attn_pool.load_state_dict(head_state['attn_pool'])
            if 'ema_decay_logit' in head_state and hasattr(model, 'ema_decay_logit'):
                model.ema_decay_logit.data = head_state['ema_decay_logit']
            if 'block_attn' in head_state and hasattr(model, 'block_attn'):
                model.block_attn.load_state_dict(head_state['block_attn'])
                model.block_query.data = head_state['block_query']
            logger.info(f"Loaded retrieval head from {head_path}")
        except FileNotFoundError:
            pass

        return model

    def save(self, output_dir: str):
        """Save model."""
        os.makedirs(output_dir, exist_ok=True)

        self.encoder.save_pretrained(output_dir)

        # Save retriever head weights (aggregation layers only)
        head_state = {}
        if hasattr(self, 'attn_pool'):
            head_state['attn_pool'] = self.attn_pool.state_dict()
        if hasattr(self, 'ema_decay_logit'):
            head_state['ema_decay_logit'] = self.ema_decay_logit.data
        if hasattr(self, 'block_attn'):
            head_state['block_attn'] = self.block_attn.state_dict()
            head_state['block_query'] = self.block_query.data

        if head_state:
            torch.save(head_state, f"{output_dir}/retrieval_head.pt")

        config = {
            'encoding_mode': self.encoding_mode,
            'block_schedule': {
                'block_length': self.block_schedule.block_length,
                'num_steps_per_block': self.block_schedule.num_steps_per_block,
                'enable_t2t': self.block_schedule.enable_t2t,
            },
            'block_aggregation': self.block_aggregation,
            'pooling': self.pooling,
            'normalize': self.normalize,
            'mask_token_id': self.mask_token_id,
            'num_repr_tokens': self.num_repr_tokens,
            'num_denoise_steps': self.num_denoise_steps,
            'query_prefix': self.query_prefix,
            'query_suffix': self.query_suffix,
            'passage_prefix': self.passage_prefix,
            'passage_suffix': self.passage_suffix,
        }
        with open(f"{output_dir}/llada2_retrieval_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved to {output_dir}")
