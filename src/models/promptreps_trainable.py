"""
PromptReps — fine-tunable autoregressive variant (LLaMA3 / Qwen2.5).

Contrastively fine-tunable counterpart to src/models/promptreps.py.
Same retrieval prompt and per-token readout as the zero-shot
PromptRepsRetriever, but with LoRA-trainable parameters and the
single-pass bidirectional readout block that paper §3.4 trains.

Sequence layout:
  causal:
    [prefix][text][suffix]
  bidirectional K>1:
    [prefix][text][suffix][pool]^{K-1}

Where suffix ends with a closing " token (the "quotation_emb" position).

K == 1: single forward pass, extract at " position.
K >  1, causal: autoregressive generation of K tokens.
  At each step, greedy-decodes next token and collects hidden state + logits.
  Uses KV cache for efficiency. Always generates exactly K-1 steps (no early stop).

K >  1, bidirectional: single forward pass over a pre-appended K-token
readout block [", pool_1, ..., pool_{K-1}] with full attention among those
positions (LLM2Vec-inspired attention, PromptReps-style readout).

Loss:
  K == 1: InfoNCE(repr_hidden / T) + sparse_weight × InfoNCE(sparse)
  K >  1: ColBERT MaxSim InfoNCE(repr_hidden / K / T)
          + sparse_weight × InfoNCE(sparse)

For K > 1:
  causal:
    - repr_hidden [B, K, H] = hidden states at each autoregressive generation step.
    - quotation_emb [B, H]  = hidden state at the " position (from initial forward pass).
    - sparse_acts [B, V]    = max-pool of log(1+relu(logit)) across K generation steps.
  bidirectional:
    - repr_hidden [B, K, H] = hidden states at the final K readout positions
      [", pool_1, ..., pool_{K-1}] from one full-attention pass.
    - quotation_emb [B, H]  = repr_hidden[:, 0, :]
    - sparse_acts [B, V]    = max-pool of log(1+relu(logit)) across the K readout positions.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .sparse_utils import filter_sparse

logger = logging.getLogger(__name__)

_NUM_WORDS = ['one', 'two', 'three', 'four', 'five',
              'six', 'seven', 'eight', 'nine', 'ten']


class TrainableARRetriever(nn.Module):
    """
    Trainable PromptReps-style AR retriever (llama / qwen).

    Build via from_pretrained(); use tokenize() in the data collator, then
    pass pre-tokenized tensors to forward() during training.
    """

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer,
        hidden_size: int,
        query_prefix_ids: List[int],
        query_suffix_ids: List[int],
        passage_prefix_ids: List[int],
        passage_suffix_ids: List[int],
        pool_token_id: int,
        max_length: int = 512,
        n_pooled_tokens: int = 1,
        temperature: float = 0.01,
        sparse_weight: float = 1.0,
        normalize: bool = True,
        flash_attn: bool = False,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size

        self.register_buffer('_dummy', torch.zeros(1))
        self._query_prefix_ids = list(query_prefix_ids)
        self._query_suffix_ids = list(query_suffix_ids)
        self._passage_prefix_ids = list(passage_prefix_ids)
        self._passage_suffix_ids = list(passage_suffix_ids)
        self.pool_token_id = pool_token_id

        self.max_length = max_length
        self.n_pooled_tokens = n_pooled_tokens
        self.temperature = temperature
        self.sparse_weight = sparse_weight
        self.dense_weight = 1.0
        self.normalize = normalize
        self.flash_attn = flash_attn
        self.bidirectional = bidirectional
        self.model_type = 'unknown'

    # ----------------------------------------------------------------
    # Build from pretrained
    # ----------------------------------------------------------------

    @staticmethod
    def _enable_bidirectional_attention(backbone):
        """Patch a causal LLM backbone to use full bidirectional attention.

        Follows the attention-side change from LLM2Vec
        (BehnamGhader et al. 2024, arXiv:2404.05961):
          1. Set is_causal=False on all attention modules — prevents PyTorch
             SDPA from generating its own internal causal mask.
          2. Replace _update_causal_mask with a version that returns an
             all-zeros additive mask (= no masking) instead of the upper-
             triangular -inf causal mask.

        Must be called after loading the backbone but before wrapping with
        LoRA / gradient checkpointing.
        """
        import types

        def _full_attention_mask(
            self, attention_mask, input_tensor, cache_position,
            past_key_values, output_attentions=False
        ):
            # Return all-zeros additive mask: every position can attend to
            # every other position (bidirectional).  Zeros rather than None
            # to remain compatible with both eager and SDPA backends.
            dtype = input_tensor.dtype
            device = input_tensor.device
            bsz, seq_len = input_tensor.shape[:2]
            # Try to honour the padding mask when provided.
            if attention_mask is not None and attention_mask.dim() == 2:
                # Convert [B, S] bool/int mask → 4D additive mask
                # 0 = attend, -inf = ignore
                expanded = attention_mask[:, None, None, :].to(dtype)
                expanded = (1.0 - expanded) * torch.finfo(dtype).min
                return expanded
            return torch.zeros(bsz, 1, seq_len, seq_len, dtype=dtype, device=device)

        inner = getattr(backbone, 'model', backbone)
        if hasattr(inner, '_update_causal_mask'):
            inner._update_causal_mask = types.MethodType(_full_attention_mask, inner)
        for module in backbone.modules():
            if hasattr(module, 'is_causal'):
                module.is_causal = False
        logger.info("AR: bidirectional attention enabled (LLM2Vec-inspired, causal mask disabled)")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        model_type: str,
        query_prompt: str,
        passage_prompt: str,
        max_length: int = 512,
        n_pooled_tokens: int = 1,
        temperature: float = 0.01,
        sparse_weight: float = 1.0,
        normalize: bool = True,
        gradient_checkpointing: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        device_map=None,
        bidirectional: bool = False,
    ) -> 'TrainableARRetriever':
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Always left-pad: last real token (the '"') is always at position -1
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        common_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
        if device_map is not None:
            common_kw['device_map'] = device_map
        flash_attn = False

        # Use sdpa (not flash_attention_2) — the flash_attn library's backward
        # kernel (FlashAttnVarlenFuncBackward) produces NaN for K>1 training
        # where gradients enter at multiple positions.  PyTorch's native sdpa
        # can still dispatch to flash kernels but has a stable backward.
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation='sdpa', **common_kw)
        logger.info("AR: using SDPA attention")

        if bidirectional:
            cls._enable_bidirectional_attention(backbone)

        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            backbone = get_peft_model(backbone, lora_cfg)
            backbone.print_trainable_parameters()
        if gradient_checkpointing:
            backbone.enable_input_require_grads()
            backbone.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Pool token: the token appended K times for multi-vector repr.
        # Use EOS as a neutral pooling token.
        pool_token_id = tokenizer.eos_token_id

        # Build prompt IDs
        q_yaml = cls._load_yaml(query_prompt)
        p_yaml = cls._load_yaml(passage_prompt)
        if n_pooled_tokens > 1:
            # Adapt prompts to indicate K-word output
            q_yaml = dict(q_yaml)
            q_yaml['user_suffix'] = cls._adapt_for_k(
                q_yaml.get('user_suffix', ''), n_pooled_tokens)
            q_yaml['assistant_prefix'] = cls._adapt_for_k(
                q_yaml.get('assistant_prefix', ''), n_pooled_tokens)
            p_yaml = dict(p_yaml)
            p_yaml['user_suffix'] = cls._adapt_for_k(
                p_yaml.get('user_suffix', ''), n_pooled_tokens)
            p_yaml['assistant_prefix'] = cls._adapt_for_k(
                p_yaml.get('assistant_prefix', ''), n_pooled_tokens)

        q_prefix_ids, q_suffix_ids = cls._build_prompt_ids(tokenizer, q_yaml)
        p_prefix_ids, p_suffix_ids = cls._build_prompt_ids(tokenizer, p_yaml)

        logger.info(f"Query prompt: {len(q_prefix_ids)} prefix + {len(q_suffix_ids)} suffix tokens")
        logger.info(f"Passage prompt: {len(p_prefix_ids)} prefix + {len(p_suffix_ids)} suffix tokens")
        logger.info(f"n_pooled_tokens={n_pooled_tokens}")
        if sparse_weight > 0:
            logger.info(f"Sparse: sparse_weight={sparse_weight}")

        model = cls(
            backbone=backbone,
            tokenizer=tokenizer,
            hidden_size=backbone.config.hidden_size,
            query_prefix_ids=q_prefix_ids,
            query_suffix_ids=q_suffix_ids,
            passage_prefix_ids=p_prefix_ids,
            passage_suffix_ids=p_suffix_ids,
            pool_token_id=pool_token_id,
            max_length=max_length,
            n_pooled_tokens=n_pooled_tokens,
            temperature=temperature,
            sparse_weight=sparse_weight,
            normalize=normalize,
            flash_attn=flash_attn,
            bidirectional=bidirectional,
        )
        model.model_type = model_type
        return model

    # ----------------------------------------------------------------
    # Prompt helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: str) -> dict:
        import yaml
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt YAML not found: {path}")
        return yaml.safe_load(p.read_text())

    @staticmethod
    def _adapt_for_k(text: str, k: int) -> str:
        if k <= 1 or not text:
            return text
        count = _NUM_WORDS[k - 1] if k <= len(_NUM_WORDS) else str(k)
        result = re.sub(
            r'\b(?:' + '|'.join(_NUM_WORDS) + r')\b(\s+words?)',
            lambda m: f'{count} words', text,
        )
        return re.sub(r'\bword is\b', 'words are', result)

    @staticmethod
    def _build_prompt_ids(tokenizer, yaml_dict: dict) -> Tuple[List[int], List[int]]:
        system = yaml_dict.get('system', '')
        user_prefix = yaml_dict.get('user_prefix', '')
        user_suffix = yaml_dict.get('user_suffix', '')
        assistant_prefix = yaml_dict.get('assistant_prefix', '')

        SENTINEL = "XSENTINELX"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_prefix + SENTINEL + user_suffix})

        full_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pre_str, post_str = full_str.split(SENTINEL, 1)
        prefix_ids = tokenizer.encode(pre_str, add_special_tokens=False)
        suffix_ids = tokenizer.encode(post_str + assistant_prefix, add_special_tokens=False)
        return prefix_ids, suffix_ids

    # ----------------------------------------------------------------
    # Tokenization
    # ----------------------------------------------------------------

    def tokenize(
        self, texts: List[str], is_query: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_ids, attention_mask) on CPU.

        Causal K=1  : [prefix][text][suffix]
        Causal K>1  : [prefix][text][suffix]  — K-1 tokens generated autoregressively
        Bidir  K=1  : [prefix][text][suffix]
        Bidir  K>1  : [prefix][text][suffix][pool]*{K-1}  — all K tokens present upfront,
                      attended to bidirectionally in a single forward pass (like diffusion MASKs)
        """
        prefix_ids = self._query_prefix_ids if is_query else self._passage_prefix_ids
        suffix_ids = self._query_suffix_ids if is_query else self._passage_suffix_ids
        K = self.n_pooled_tokens
        pool_tail = [self.pool_token_id] * (K - 1) if (self.bidirectional and K > 1) else []
        max_text_len = self.max_length - len(prefix_ids) - len(suffix_ids)

        enc = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_text_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        enc['input_ids'] = [
            prefix_ids + ids + suffix_ids + pool_tail
            for ids in enc['input_ids']
        ]
        collated = self.tokenizer.pad(
            enc,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return collated['input_ids'], collated['attention_mask']

    # ----------------------------------------------------------------
    # Backbone forward
    # ----------------------------------------------------------------

    def _fwd(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        need_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Causal LM forward pass. Returns (hidden [B,L,H], logits [B,L,V] or None).

        Returns native dtype (bf16). Callers convert only the small slices they need.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        logits = outputs.logits if need_logits else None
        return hidden, logits

    # ----------------------------------------------------------------
    # Encode (inference)
    # ----------------------------------------------------------------

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_query: bool = False,
        compute_sparse: Optional[bool] = None,
        content_token_ids: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts.

        K == 1: Single forward pass.
          quotation_emb = hidden at last real token (" position).
          sparse_acts   = logits at that position.
          repr_hidden   = quotation_emb unsqueezed to [B, 1, H].

        K > 1, bidirectional: Single forward pass over the pre-appended
          readout block [", pool_1, ..., pool_{K-1}] with full attention.
          repr_hidden   = hidden states of those final K positions.
          quotation_emb = hidden state at the " position (first of the K).
          sparse_acts   = max-pool of log(1+relu(logit)) across those K positions.

        K > 1, causal: Autoregressive generation (matching PromptReps).
          Generates K tokens greedily with KV cache. At each step, collects
          the hidden state and logits. Stops early if closing '"' is generated.
          quotation_emb = hidden at " position (from initial forward pass).
          repr_hidden   = [B, K, H] hidden states at each generated position.
          sparse_acts   = max-pool of log(1+relu(logit)) across K positions.
        """
        K = self.n_pooled_tokens
        B = input_ids.size(0)
        device = input_ids.device

        if compute_sparse is None:
            need_sparse = self.sparse_weight > 0
        else:
            need_sparse = compute_sparse

        # ── K == 1: single forward pass ───────────────────────────────────────
        if K <= 1:
            hidden, logits = self._fwd(input_ids, attention_mask, need_logits=need_sparse)
            # With left-padding, the last real token (closing ") is always
            # at position L-1 (content is right-aligned).
            # Convert only the small slice [B, H] to float32, not full [B, L, H].
            quotation_emb = hidden[:, -1, :].float()
            repr_hidden = quotation_emb.unsqueeze(1)   # [B, 1, H]
            sparse_max = None
            if need_sparse and logits is not None:
                sparse_max = torch.log(1.0 + torch.relu(logits[:, -1, :].float()))

        # ── K > 1, bidirectional: single encoder-style pass ─────────────────
        elif self.bidirectional:
            hidden, logits = self._fwd(input_ids, attention_mask, need_logits=need_sparse)
            # tokenize() guarantees that for bidirectional K>1 the final K
            # positions are exactly [", pool_1, ..., pool_{K-1}].
            last_hidden = hidden[:, -K:, :].float()
            quotation_emb = last_hidden[:, 0, :]
            repr_hidden = last_hidden
            sparse_max = None
            if need_sparse and logits is not None:
                sparse_logits = logits[:, -K:, :].float()
                sparse_max = torch.log(1.0 + torch.relu(sparse_logits)).max(dim=1).values

        # ── K > 1, causal: autoregressive generation ────────────────────────
        # Matches original PromptReps multi_reps layout:
        #   repr_hidden[:, 0, :] = hidden state at '"' (quotation mark)
        #   repr_hidden[:, k, :] = hidden state at (k-1)-th generated token
        # So K vectors = 1 prompt pass + (K-1) generation steps.
        #
        # Two modes:
        # - Inference (no grad): KV cache for efficiency
        # - Training (grad checkpointing): full forward passes without KV cache
        else:
            _use_kv_cache = not self.training

            hidden_size = self.backbone.config.hidden_size
            repr_hidden = torch.zeros(B, K, hidden_size, device=device)
            sparse_max: Optional[torch.Tensor] = None

            full_ids = input_ids      # [B, L]
            full_mask = attention_mask  # [B, L]

            if _use_kv_cache:
                # ── Inference path: KV cache (fast) ──────────────────────
                outputs = self.backbone(
                    input_ids=full_ids, attention_mask=full_mask,
                    output_hidden_states=True, return_dict=True, use_cache=True)
                quotation_emb = outputs.hidden_states[-1][:, -1, :].float()
                past_kv = outputs.past_key_values

                # Position 0: '"' hidden state (same as quotation_emb)
                repr_hidden[:, 0, :] = quotation_emb

                step_logits = outputs.logits[:, -1, :].float()
                next_token = step_logits.argmax(dim=-1).unsqueeze(1)
                if need_sparse:
                    sparse_max = torch.log(1.0 + torch.relu(step_logits))

                # Generate K-1 tokens, placing at positions 1..K-1
                for step in range(K - 1):
                    full_mask = torch.cat([full_mask, torch.ones(B, 1, device=device, dtype=full_mask.dtype)], dim=1)
                    outputs = self.backbone(
                        input_ids=next_token, attention_mask=full_mask,
                        output_hidden_states=True, return_dict=True,
                        use_cache=True, past_key_values=past_kv)
                    past_kv = outputs.past_key_values
                    repr_hidden[:, step + 1, :] = outputs.hidden_states[-1][:, -1, :].float()

                    step_logits = outputs.logits[:, -1, :].float()
                    next_token = step_logits.argmax(dim=-1).unsqueeze(1)
                    if need_sparse:
                        act = torch.log(1.0 + torch.relu(step_logits))
                        sparse_max = torch.max(sparse_max, act)
            else:
                # ── Training path: generate tokens then single forward w/ grad ──
                # Old approach ran K full backbone calls with grad, creating a
                # backward graph K×32 layers deep.  bf16 gradient overflow in
                # that deep graph produced NaN weights after the first update.
                #
                # Fix: causal attention means hidden[pos] is independent of
                # future tokens, so one forward pass on the extended sequence
                # gives identical representations to K separate calls.

                # 1) Autoregressively generate K-1 tokens
                #    argmax is non-differentiable so these forward passes
                #    don't connect to the loss graph and get garbage-collected.
                for _ in range(K - 1):
                    gen_out = self.backbone(
                        input_ids=full_ids, attention_mask=full_mask,
                        output_hidden_states=False, return_dict=True)
                    next_token = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    del gen_out  # free computation graph immediately
                    full_ids = torch.cat([full_ids, next_token], dim=1)
                    full_mask = torch.cat([
                        full_mask,
                        torch.ones(B, 1, device=device, dtype=full_mask.dtype),
                    ], dim=1)

                # 2) Single forward pass with grad on the extended sequence
                outputs = self.backbone(
                    input_ids=full_ids, attention_mask=full_mask,
                    output_hidden_states=True, return_dict=True)

                # Last K positions: [", tok1, tok2, …, tok_{K-1}]
                last_hidden = outputs.hidden_states[-1]
                quotation_emb = last_hidden[:, -K, :].float()
                repr_hidden = last_hidden[:, -K:, :].float()   # [B, K, H]

                if need_sparse:
                    sparse_logits = outputs.logits[:, -K:, :].float()   # [B, K, V]
                    sparse_max = torch.log(1.0 + torch.relu(sparse_logits)).max(dim=1).values

        if self.normalize:
            quotation_emb = F.normalize(quotation_emb, p=2, dim=-1)
            repr_hidden = F.normalize(repr_hidden, p=2, dim=-1)

        result: Dict[str, torch.Tensor] = {
            'repr_hidden': repr_hidden,
            'quotation_emb': quotation_emb,
        }
        if sparse_max is not None:
            if content_token_ids is not None:
                sparse_max = filter_sparse(sparse_max, content_token_ids)
            result['sparse_acts'] = sparse_max

        return result

    # ----------------------------------------------------------------
    # MaxSim (ColBERT-style)
    # ----------------------------------------------------------------

    @staticmethod
    def maxsim(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """q: [B_q, k, H], p: [B_p, k, H]  →  [B_q, B_p] MaxSim scores."""
        sims = torch.einsum('ikh,jlh->ijkl', q, p)
        return sims.max(dim=-1).values.sum(dim=-1)

    # ----------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------

    def compute_loss(
        self,
        q_repr: Dict[str, torch.Tensor],
        p_repr: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Dense loss:
          - K == 1: dense_loss = InfoNCE on repr_hidden (single-vector dot product)
          - K > 1: dense_loss = ColBERT MaxSim InfoNCE on repr_hidden
        Sparse InfoNCE (if sparse_weight > 0): raw dot product + temperature.
        """
        device = labels.device
        K = q_repr['repr_hidden'].size(1)

        total_loss = torch.tensor(0.0, device=device)

        if K > 1:
            # ColBERT MaxSim on repr_hidden
            colbert_scores = self.maxsim(q_repr['repr_hidden'], p_repr['repr_hidden'])
            colbert_loss = F.cross_entropy(colbert_scores / K / self.temperature, labels)
            total_loss = self.dense_weight * colbert_loss
        else:
            # K==1: single-vector dot product on repr_hidden
            q_vec = q_repr['repr_hidden'].squeeze(1)  # [B, H]
            p_vec = p_repr['repr_hidden'].squeeze(1)  # [B, H]
            dense_scores = q_vec @ p_vec.T
            dense_loss = F.cross_entropy(dense_scores / self.temperature, labels)
            total_loss = self.dense_weight * dense_loss

        # Sparse InfoNCE — raw dot product, same as inference.
        # Content-token filtering (PromptReps-style) applied in forward() keeps
        # scores small (~10-100), so no clamping or normalization needed.
        sparse_loss = torch.tensor(0.0, device=device)
        if (self.sparse_weight > 0
                and 'sparse_acts' in q_repr
                and 'sparse_acts' in p_repr):
            sparse_scores = q_repr['sparse_acts'] @ p_repr['sparse_acts'].T
            sparse_loss = F.cross_entropy(sparse_scores, labels)
            total_loss = total_loss + self.sparse_weight * sparse_loss

        dense_term = colbert_loss if K > 1 else dense_loss
        return {
            'loss': total_loss,
            'loss_dense': (self.dense_weight * dense_term).detach(),
            'loss_sparse': sparse_loss.detach(),
        }

    # ----------------------------------------------------------------
    # Cross-GPU negative sharing
    # ----------------------------------------------------------------

    @staticmethod
    def _dist_gather(t: torch.Tensor) -> torch.Tensor:
        """All-gather tensors across GPUs with gradient passthrough."""
        if not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
            return t
        gathered = [torch.zeros_like(t) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, t.contiguous())
        gathered[torch.distributed.get_rank()] = t
        return torch.cat(gathered, dim=0)

    def _gather_repr(self, repr_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Gather all representation tensors across GPUs."""
        out = {}
        for k, v in repr_dict.items():
            out[k] = self._dist_gather(v)
        return out

    # ----------------------------------------------------------------
    # HF Trainer-compatible forward
    # ----------------------------------------------------------------

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        passage_input_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
        query_content_ids: Optional[List] = None,
        passage_content_ids: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        passages layout: [pos_0, neg_0_0, …, neg_0_M, pos_1, …]
        Positive for query i is at index i * (1 + n_neg).
        """
        B_q = query_input_ids.size(0)
        B_p = passage_input_ids.size(0)
        assert B_p % B_q == 0
        n_paq = B_p // B_q

        q_repr = self.encode(query_input_ids, query_attention_mask, is_query=True,
                             compute_sparse=self.sparse_weight > 0,
                             content_token_ids=query_content_ids)
        p_repr = self.encode(passage_input_ids, passage_attention_mask, is_query=False,
                             content_token_ids=passage_content_ids)

        # Cross-GPU negative sharing: gather all representations
        q_repr = self._gather_repr(q_repr)
        p_repr = self._gather_repr(p_repr)

        # Recompute labels after gather (each GPU's positives are offset)
        B_q_all = q_repr['quotation_emb'].size(0)
        B_p_all = p_repr['quotation_emb'].size(0)
        n_paq_all = B_p_all // B_q_all
        labels = torch.arange(B_q_all, device=query_input_ids.device) * n_paq_all

        loss_dict = self.compute_loss(q_repr, p_repr, labels)

        # Scale loss to counter DDP averaging
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            loss_dict['loss'] = loss_dict['loss'] * torch.distributed.get_world_size()

        return loss_dict

    # ----------------------------------------------------------------
    # Save / load
    # ----------------------------------------------------------------

    def _save_config(self, output_dir: str):
        import json, os
        config = {
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'max_length': self.max_length,
            'n_pooled_tokens': self.n_pooled_tokens,
            'pool_token_id': self.pool_token_id,
            'temperature': self.temperature,
            'sparse_weight': self.sparse_weight,
            'normalize': self.normalize,
            'bidirectional': self.bidirectional,
            'query_prefix_ids': self._query_prefix_ids,
            'query_suffix_ids': self._query_suffix_ids,
            'passage_prefix_ids': self._passage_prefix_ids,
            'passage_suffix_ids': self._passage_suffix_ids,
        }
        with open(os.path.join(output_dir, 'ar_retriever_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def save(self, output_dir: str):
        import os
        os.makedirs(output_dir, exist_ok=True)
        backbone = self.backbone
        if hasattr(backbone, 'save_pretrained'):
            backbone.save_pretrained(output_dir)
        else:
            backbone.base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self._save_config(output_dir)
        logger.info(f"Saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str, **fallback_kwargs) -> 'TrainableARRetriever':
        """Load a fine-tuned TrainableARRetriever from a saved checkpoint directory.

        Handles:
        - Final model dirs (ar_retriever_config.json present)
        - Intermediate checkpoints (checkpoint-N/) — searches parent dir for config
        - No config at all — falls back to from_pretrained() using adapter_config.json
          + fallback_kwargs (model_type, query_prompt, passage_prompt, n_pooled_tokens)
        """
        import json, os
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Search for ar_retriever_config.json: checkpoint dir → parent dir
        config_path = os.path.join(model_dir, 'ar_retriever_config.json')
        if not os.path.exists(config_path):
            parent = os.path.dirname(os.path.normpath(model_dir))
            config_path = os.path.join(parent, 'ar_retriever_config.json')

        config = None
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")

        # Load backbone (LoRA adapter or full model)
        common_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
        flash_attn = False
        tokenizer = None

        # Resolve base model name: adapter_config.json → config → fallback kwarg → dir name
        _ORIGINAL_MODELS = {
            'llama': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'qwen': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen25': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen3': 'Qwen/Qwen3-8B',
        }
        adapter_config_path = os.path.join(model_dir, 'adapter_config.json')
        base_model_name = None

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path) as f:
                base_model_name = json.load(f)['base_model_name_or_path']
        elif config is not None:
            base_model_name = _ORIGINAL_MODELS.get(config.get('model_type'))
        else:
            _mt = fallback_kwargs.get('model_type')
            if _mt:
                base_model_name = _ORIGINAL_MODELS.get(_mt)

        if os.path.exists(adapter_config_path):
            # Standard PEFT adapter loading — use sdpa (flash_attention_2 backward
            # produces NaN for K>1 and crashes on some BEIR sequence lengths)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, attn_implementation='sdpa', **common_kw)
            from peft import PeftModel
            backbone = PeftModel.from_pretrained(base_model, model_dir)
            backbone = backbone.merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            logger.info(f"Loaded LoRA adapter from {model_dir} (base: {base_model_name})")
        elif base_model_name:
            # DeepSpeed checkpoint: full state dict with PEFT keys.
            # No adapter_config.json, no tokenizer — load everything from HuggingFace base.
            from safetensors.torch import load_file

            # Load state dict
            weight_file = Path(model_dir) / 'model.safetensors'
            bin_files = sorted(Path(model_dir).glob('model-*.safetensors'))
            state_dict = None
            if weight_file.exists():
                state_dict = load_file(str(weight_file))
            elif bin_files:
                state_dict = {}
                for f in bin_files:
                    state_dict.update(load_file(str(f)))

            # Detect LoRA rank
            detected_lora_rank = 0
            if state_dict:
                for k, v in state_dict.items():
                    if 'lora_A.default.weight' in k:
                        detected_lora_rank = v.shape[0]
                        break

            # Resolve prompt YAML paths from model_type
            _PROMPT_DIRS = {'llama': 'llama3', 'qwen': 'qwen', 'qwen25': 'qwen', 'qwen3': 'qwen'}
            model_type = (config or {}).get('model_type') or fallback_kwargs.get('model_type', 'llama')
            _prompt_dir = _PROMPT_DIRS.get(model_type, model_type)
            _q_prompt = fallback_kwargs.get('query_prompt') or f'prompts/{_prompt_dir}/query_prompt.yaml'
            _p_prompt = fallback_kwargs.get('passage_prompt') or f'prompts/{_prompt_dir}/passage_prompt.yaml'

            n_pooled = fallback_kwargs.get('n_pooled_tokens', 1)
            model_tmp = cls.from_pretrained(
                model_name=base_model_name,
                model_type=model_type,
                query_prompt=_q_prompt,
                passage_prompt=_p_prompt,
                n_pooled_tokens=n_pooled,
                lora_rank=detected_lora_rank,
                gradient_checkpointing=False,
                device_map='auto',
            )
            # Load full state dict (backbone.base_model.model.* keys)
            if state_dict:
                missing, unexpected = model_tmp.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded DeepSpeed checkpoint from {model_dir} "
                            f"(lora_rank={detected_lora_rank}, "
                            f"{len(missing)} missing, {len(unexpected)} unexpected)")
            backbone = model_tmp.backbone
            # Merge LoRA weights into base model for faster inference
            if detected_lora_rank > 0 and hasattr(backbone, 'merge_and_unload'):
                backbone = backbone.merge_and_unload()
                logger.info("Merged LoRA adapters for inference")
            tokenizer = model_tmp.tokenizer
            flash_attn = model_tmp.flash_attn
            # Use the config built by from_pretrained if we don't have one
            if config is None:
                config = {
                    'model_type': model_type,
                    'hidden_size': backbone.config.hidden_size,
                    'max_length': fallback_kwargs.get('max_length', 256),
                    'n_pooled_tokens': n_pooled,
                    'pool_token_id': tokenizer.eos_token_id,
                    'temperature': fallback_kwargs.get('temperature', 0.01),
                    'sparse_weight': fallback_kwargs.get('sparse_weight', 1.0),
                    'normalize': fallback_kwargs.get('normalize', True),
                    'query_prefix_ids': model_tmp._query_prefix_ids,
                    'query_suffix_ids': model_tmp._query_suffix_ids,
                    'passage_prefix_ids': model_tmp._passage_prefix_ids,
                    'passage_suffix_ids': model_tmp._passage_suffix_ids,
                }
        else:
            backbone = AutoModelForCausalLM.from_pretrained(
                model_dir, attn_implementation='sdpa', **common_kw)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            logger.info(f"Loaded full model from {model_dir}")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Defer padding_side until we know n_pooled_tokens (set below after config is resolved)

        # If no config found, build from fallback kwargs (same as from_pretrained)
        if config is None:
            model_type = fallback_kwargs.get('model_type')
            query_prompt = fallback_kwargs.get('query_prompt')
            passage_prompt = fallback_kwargs.get('passage_prompt')
            if not model_type or not query_prompt or not passage_prompt:
                raise FileNotFoundError(
                    f"No ar_retriever_config.json in {model_dir} or parent dir, "
                    f"and no fallback kwargs (model_type, query_prompt, passage_prompt) provided.")

            n_pooled = fallback_kwargs.get('n_pooled_tokens', 1)
            q_yaml = cls._load_yaml(query_prompt)
            p_yaml = cls._load_yaml(passage_prompt)
            if n_pooled > 1:
                q_yaml = dict(q_yaml)
                q_yaml['user_suffix'] = cls._adapt_for_k(q_yaml.get('user_suffix', ''), n_pooled)
                q_yaml['assistant_prefix'] = cls._adapt_for_k(q_yaml.get('assistant_prefix', ''), n_pooled)
                p_yaml = dict(p_yaml)
                p_yaml['user_suffix'] = cls._adapt_for_k(p_yaml.get('user_suffix', ''), n_pooled)
                p_yaml['assistant_prefix'] = cls._adapt_for_k(p_yaml.get('assistant_prefix', ''), n_pooled)

            q_prefix_ids, q_suffix_ids = cls._build_prompt_ids(tokenizer, q_yaml)
            p_prefix_ids, p_suffix_ids = cls._build_prompt_ids(tokenizer, p_yaml)

            config = {
                'model_type': model_type,
                'hidden_size': backbone.config.hidden_size,
                'max_length': fallback_kwargs.get('max_length', 256),
                'n_pooled_tokens': n_pooled,
                'pool_token_id': tokenizer.eos_token_id,
                'temperature': fallback_kwargs.get('temperature', 0.01),
                'sparse_weight': fallback_kwargs.get('sparse_weight', 1.0),
                'normalize': fallback_kwargs.get('normalize', True),
                'query_prefix_ids': q_prefix_ids,
                'query_suffix_ids': q_suffix_ids,
                'passage_prefix_ids': p_prefix_ids,
                'passage_suffix_ids': p_suffix_ids,
            }
            logger.info(f"No config file — built from fallback kwargs (model_type={model_type}, K={n_pooled})")

        bidir = config.get('bidirectional', False)
        if bidir:
            cls._enable_bidirectional_attention(backbone)

        model = cls(
            backbone=backbone,
            tokenizer=tokenizer,
            hidden_size=config['hidden_size'],
            query_prefix_ids=config['query_prefix_ids'],
            query_suffix_ids=config['query_suffix_ids'],
            passage_prefix_ids=config['passage_prefix_ids'],
            passage_suffix_ids=config['passage_suffix_ids'],
            pool_token_id=config['pool_token_id'],
            max_length=config['max_length'],
            n_pooled_tokens=config['n_pooled_tokens'],
            temperature=config.get('temperature', 0.02),
            sparse_weight=config.get('sparse_weight', 1.0),
            normalize=config.get('normalize', True),
            flash_attn=flash_attn,
            bidirectional=bidir,
        )
        model.model_type = config.get('model_type', 'unknown')
        # Always left-pad: last real token (the '"') is always at position -1
        tokenizer.padding_side = 'left'
        return model

    @property
    def config(self):
        return self.backbone.config

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)
