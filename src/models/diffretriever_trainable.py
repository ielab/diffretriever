"""
Trainable Diffusion Retriever — fine-tunable Dream/LLaDA2 model.

Supports Dream, LLaDA1/1.5, LLaDA2 backbones via backbone_adapters.py.

Encoding:
  steps=1 (fast): Single forward pass over [prefix][text][suffix][MASK×K].
    repr_hidden[:, s, :] from MASK position s; quotation_emb from the
    closing " token.
  steps>1 (rich): Iterative denoising loop with mixed representations.
    At each step, gen positions already decoded in a prior step contribute
    their frozen hidden state (no gradient); positions still MASK contribute
    the current step's hidden (with gradient).
    Uniform LLaDA-style unmasking schedule: n_per_step = K // n_steps
    tokens are decoded per step.

Loss (training):
  K > 1:   ColBERT MaxSim InfoNCE over all K mixed vectors.
  steps>1: loss computed only at the final denoising step so gradient
           flows from the fully-contextualized representation without dilution.
           If progressive_step_weight > 0, retrieval loss is also applied at
           each intermediate step with linearly increasing weight (t/T),
           giving direct supervision to early denoising steps.
  Optional: sparse InfoNCE + FLOPS L1 regularization.

At eval time the same encode() output (repr_hidden, quotation_emb,
sparse_acts) supports all zero-shot retrieval modes: single_dense,
multi_dense_k*, multi-denoise-step variants, and sparse versions of each.
"""

import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .sparse_utils import filter_sparse
from .backbone_adapters import get_adapter, BackboneAdapter

logger = logging.getLogger(__name__)

_NUM_WORDS = ['one', 'two', 'three', 'four', 'five',
              'six', 'seven', 'eight', 'nine', 'ten']


class TrainableDiffusionRetriever(nn.Module):
    """
    Trainable Diffusion Retriever.

    Build via from_backbone(); use tokenize() in your data collator, then pass
    pre-tokenized tensors to forward() during training.
    """

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer,
        mask_token_id: int,
        hidden_size: int,
        query_prefix_ids: List[int],
        query_suffix_ids: List[int],
        passage_prefix_ids: List[int],
        passage_suffix_ids: List[int],
        max_length: int = 512,
        n_gen_tokens: int = 4,
        n_gen_q_tokens: Optional[int] = None,    # asymmetric K_q (defaults to n_gen_tokens)
        n_gen_p_tokens: Optional[int] = None,    # asymmetric K_p (defaults to n_gen_tokens)
        temperature: float = 0.01,
        num_denoise_steps: int = 4,
        sparse_weight: float = 1.0,
        normalize: bool = True,
        flash_attn: bool = False,
        use_eos: bool = False,
        # K-adapter (joint training of K-router + retriever).  When enabled,
        # the model encodes at K_max and at training time computes per-cell
        # InfoNCE losses; a tiny MLP head maps the query's quotation_emb to a
        # softmax over K-cell choices and is supervised by KL(teacher || head)
        # where teacher = softmax(-per_cell_loss / τ_T).  See compute_loss().
        use_k_adapter: bool = False,
        adapter_weight: float = 1.0,
        teacher_temperature: float = 1.0,
        k_adapter_options: Optional[Tuple[int, ...]] = None,
        # Two-stage K-pre-encoder (true encoding savings via per-item K).
        # When enabled, a tiny MLP runs over the embedding-layer output and
        # predicts K_q (for queries) and K_p (for passages) per item.  Each
        # item's input is sliced to its predicted K (removing K_max-K MASK
        # tokens), the batch is padded to max-K-in-batch, and main encoder
        # runs at variable length.  See _forward_with_k_pre_encoder().
        use_k_pre_encoder: bool = False,
        gumbel_temperature: float = 1.0,
        k_cost_lambda: float = 0.001,
        k_pre_encoder_options: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size

        # Prompt IDs (cached lists, not parameters)
        self.register_buffer('_dummy', torch.zeros(1))  # for .device
        self._query_prefix_ids = list(query_prefix_ids)
        self._query_suffix_ids = list(query_suffix_ids)
        self._passage_prefix_ids = list(passage_prefix_ids)
        self._passage_suffix_ids = list(passage_suffix_ids)

        self.max_length = max_length
        # Per-side K.  When the user passes only `n_gen_tokens`, both sides
        # use it (symmetric — original behaviour).  Pass `n_gen_q_tokens` /
        # `n_gen_p_tokens` to train asymmetric (e.g. K_q=4, K_p=16).
        self.n_gen_tokens = n_gen_tokens                   # legacy, kept for back-compat
        self.n_gen_q_tokens = n_gen_q_tokens if n_gen_q_tokens is not None else n_gen_tokens
        self.n_gen_p_tokens = n_gen_p_tokens if n_gen_p_tokens is not None else n_gen_tokens
        self.use_eos = use_eos  # kept for config compat; _n_tail is authoritative
        # _n_tail is the structural tail (quote + chat_end + EOS); it's the
        # same regardless of K_q or K_p as long as at least one side has K>0.
        max_k = max(self.n_gen_q_tokens, self.n_gen_p_tokens)
        self._n_tail = 3 if max_k > 0 else 0
        self._tail_ids: List[int] = self._build_tail_ids()  # cached at init
        self.temperature = temperature
        # Denoise step count is capped by the larger K (the side with more
        # mask tokens has more positions to iteratively decode).
        self.num_denoise_steps = min(num_denoise_steps, max_k)
        self.sparse_weight = sparse_weight
        self.dense_weight = 1.0
        self.normalize = normalize
        self.flash_attn = flash_attn
        self.model_type = 'unknown'  # set by from_backbone() / load()
        self.adapter: Optional[BackboneAdapter] = None   # set by factory methods
        self._hook_registered = False                    # set by _setup_hook()

        # Auxiliary losses (activated by setting weight > 0)
        self.denoising_weight = 0.0    # diffusion-native masked text denoising
        self.diversity_weight = 0.0    # explicit multi-vector diversity
        self.denoise_mask_ratio = 0.15 # fraction of text tokens to mask for denoising
        self.progressive_step_weight = 0.0  # progressive step supervision (multi-step only)
        self.use_fresh_final = False  # "fresh" ablation: use current hidden for ALL K at final step
        self.soft_denoising = False   # soft-token multi-step: differentiable embeddings instead of hard tokens
        self.soft_temperature = 1.0   # temperature for softmax in soft-token denoising
        self.corruption_rate = 0.0    # max text corruption rate for denoising-conditioned training (0 = off)
        self.debug_dense_metrics = False
        self.debug_compare_hidden_once = False
        self._debug_hidden_compared = False

        # ── K-adapter (joint K-router + retriever) ─────────────────────────
        self.use_k_adapter = bool(use_k_adapter)
        self.adapter_weight = float(adapter_weight)
        self.teacher_temperature = float(teacher_temperature)
        # Default K options: factors of K_max that fall within the active K
        # range.  E.g., K_max=16 → {1,2,4,8,16}; K_max=4 → {1,2,4}.
        if k_adapter_options is None:
            k_max_active = max(self.n_gen_q_tokens, self.n_gen_p_tokens)
            base = (1, 2, 4, 8, 16)
            self.k_adapter_options: Tuple[int, ...] = tuple(
                k for k in base if k <= k_max_active
            )
        else:
            self.k_adapter_options = tuple(int(k) for k in k_adapter_options)
        self.n_K = len(self.k_adapter_options)

        if self.use_k_adapter:
            # Tiny MLP: hidden_size → hidden_size//4 → n_K * n_K.
            # Trained jointly with the backbone via the per-cell loss teacher.
            # Kept in fp32 to match the .float() conversion of representations
            # in encode(); HF Trainer's bf16 autocast handles mixed precision
            # at the call site without needing a manual dtype cast here.
            head_dim = max(64, hidden_size // 4)
            self.k_adapter = nn.Sequential(
                nn.Linear(hidden_size, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, self.n_K * self.n_K),
            )
            logger.info(
                f"KAdapter enabled: K options={self.k_adapter_options}, "
                f"adapter_weight={self.adapter_weight}, "
                f"teacher_temperature={self.teacher_temperature}"
            )
        else:
            self.k_adapter = None

        # ── K-pre-encoder (two-stage encoding) ─────────────────────────────
        self.use_k_pre_encoder = bool(use_k_pre_encoder)
        self.gumbel_temperature = float(gumbel_temperature)
        self.k_cost_lambda = float(k_cost_lambda)
        if k_pre_encoder_options is None:
            k_max_active_pe = max(self.n_gen_q_tokens, self.n_gen_p_tokens)
            base_pe = (1, 2, 4, 8, 16)
            self.k_pre_encoder_options: Tuple[int, ...] = tuple(
                k for k in base_pe if k <= k_max_active_pe
            )
        else:
            self.k_pre_encoder_options = tuple(int(k) for k in k_pre_encoder_options)
        self.n_K_pe = len(self.k_pre_encoder_options)
        # Buffer of K values for differentiable expected-K (cost regularizer).
        self.register_buffer(
            'k_pe_options_tensor',
            torch.tensor(self.k_pre_encoder_options, dtype=torch.float32),
            persistent=False,
        )
        if self.use_k_pre_encoder:
            head_dim_pe = max(64, hidden_size // 4)
            # Separate q + p heads — query and passage have different optimal K
            # distributions and texts differ enough to justify dedicated heads.
            # Both heads operate on the (shared) embedding layer's output —
            # mean-pooled over real tokens.  Cheap (~1% of full forward).
            self.k_pre_encoder_q = nn.Sequential(
                nn.Linear(hidden_size, head_dim_pe),
                nn.GELU(),
                nn.Linear(head_dim_pe, self.n_K_pe),
            )
            self.k_pre_encoder_p = nn.Sequential(
                nn.Linear(hidden_size, head_dim_pe),
                nn.GELU(),
                nn.Linear(head_dim_pe, self.n_K_pe),
            )
            logger.info(
                f"K pre-encoder enabled: K options={self.k_pre_encoder_options}, "
                f"gumbel_τ={self.gumbel_temperature}, "
                f"cost_λ={self.k_cost_lambda}"
            )
        else:
            self.k_pre_encoder_q = None
            self.k_pre_encoder_p = None

    def _k(self, is_query: bool) -> int:
        """Return the appropriate per-side K.  Lets the rest of the code
        say `K = self._k(is_query)` and stay correct under both symmetric
        (K_q == K_p) and asymmetric configurations."""
        return self.n_gen_q_tokens if is_query else self.n_gen_p_tokens

    # ----------------------------------------------------------------
    # Build from pretrained backbone
    # ----------------------------------------------------------------

    @classmethod
    def from_backbone(
        cls,
        model_name: str,
        model_type: str,
        query_prompt: str,
        passage_prompt: str,
        max_length: int = 512,
        n_gen_tokens: int = 4,
        n_gen_q_tokens: Optional[int] = None,
        n_gen_p_tokens: Optional[int] = None,
        temperature: float = 0.01,
        num_denoise_steps: int = 4,
        sparse_weight: float = 1.0,
        normalize: bool = True,
        gradient_checkpointing: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        device_map=None,
        use_eos: bool = False,
        disable_hidden_hook: bool = False,
        # K-adapter passthrough
        use_k_adapter: bool = False,
        adapter_weight: float = 1.0,
        teacher_temperature: float = 1.0,
        k_adapter_options: Optional[Tuple[int, ...]] = None,
        # K pre-encoder passthrough
        use_k_pre_encoder: bool = False,
        gumbel_temperature: float = 1.0,
        k_cost_lambda: float = 0.001,
        k_pre_encoder_options: Optional[Tuple[int, ...]] = None,
    ) -> 'TrainableDiffusionRetriever':
        from transformers import AutoTokenizer

        adapter = get_adapter(model_type)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = 'left'

        backbone = adapter.load_backbone(model_name, device_map=device_map)

        if lora_rank > 0:
            from peft import get_peft_model
            lora_cfg = adapter.get_lora_config(lora_rank, lora_alpha, lora_dropout)
            backbone = get_peft_model(backbone, lora_cfg)
            backbone.print_trainable_parameters()

        if gradient_checkpointing:
            backbone.enable_input_require_grads()
            adapter.enable_gradient_checkpointing(backbone)

        mask_token_id = adapter.mask_token_id
        logger.info(f"Mask token: {model_type} → {mask_token_id}")

        # Resolve per-side K (default to symmetric n_gen_tokens).  The
        # YAML adaptation ('the relevant document is one word' → 'four
        # words') uses the per-side K so the natural-language prompt
        # matches the actual MASK count the model will see.
        k_q = n_gen_q_tokens if n_gen_q_tokens is not None else n_gen_tokens
        k_p = n_gen_p_tokens if n_gen_p_tokens is not None else n_gen_tokens

        # Build prompt token IDs (per-side adaptation)
        q_yaml = cls._load_yaml(query_prompt)
        p_yaml = cls._load_yaml(passage_prompt)
        q_yaml = dict(q_yaml)
        q_yaml['user_suffix'] = cls._adapt_for_k(q_yaml.get('user_suffix', ''), k_q)
        q_yaml['assistant_prefix'] = cls._adapt_for_k(q_yaml.get('assistant_prefix', ''), k_q)
        p_yaml = dict(p_yaml)
        p_yaml['user_suffix'] = cls._adapt_for_k(p_yaml.get('user_suffix', ''), k_p)
        p_yaml['assistant_prefix'] = cls._adapt_for_k(p_yaml.get('assistant_prefix', ''), k_p)

        q_prefix_ids, q_suffix_ids = cls._build_prompt_ids(tokenizer, q_yaml)
        p_prefix_ids, p_suffix_ids = cls._build_prompt_ids(tokenizer, p_yaml)

        logger.info(f"Query prompt: {len(q_prefix_ids)} prefix + {len(q_suffix_ids)} suffix tokens")
        logger.info(f"Passage prompt: {len(p_prefix_ids)} prefix + {len(p_suffix_ids)} suffix tokens")
        if k_q == k_p:
            logger.info(f"n_gen_tokens={k_q} (symmetric), num_denoise_steps={num_denoise_steps}")
        else:
            logger.info(f"n_gen_q_tokens={k_q}, n_gen_p_tokens={k_p} (asymmetric), "
                        f"num_denoise_steps={num_denoise_steps}")
        if sparse_weight > 0:
            logger.info(f"Sparse loss: sparse_weight={sparse_weight}")

        if num_denoise_steps is None:
            num_denoise_steps = max(k_q, k_p)

        hidden_size = backbone.config.hidden_size

        model = cls(
            backbone=backbone,
            tokenizer=tokenizer,
            mask_token_id=mask_token_id,
            hidden_size=hidden_size,
            query_prefix_ids=q_prefix_ids,
            query_suffix_ids=q_suffix_ids,
            passage_prefix_ids=p_prefix_ids,
            passage_suffix_ids=p_suffix_ids,
            max_length=max_length,
            n_gen_tokens=n_gen_tokens,
            n_gen_q_tokens=k_q,
            n_gen_p_tokens=k_p,
            temperature=temperature,
            num_denoise_steps=num_denoise_steps,
            sparse_weight=sparse_weight,
            normalize=normalize,
            flash_attn=adapter.flash_attn,
            use_eos=use_eos,
            use_k_adapter=use_k_adapter,
            adapter_weight=adapter_weight,
            teacher_temperature=teacher_temperature,
            k_adapter_options=k_adapter_options,
            use_k_pre_encoder=use_k_pre_encoder,
            gumbel_temperature=gumbel_temperature,
            k_cost_lambda=k_cost_lambda,
            k_pre_encoder_options=k_pre_encoder_options,
        )
        model.model_type = model_type
        model.adapter = adapter
        model.lora_rank = lora_rank
        model.lora_alpha = lora_alpha

        # Hook on output projection for efficient hidden state extraction
        model._last_hidden: Dict[str, torch.Tensor] = {}
        if disable_hidden_hook:
            logger.info(f"{model_type}: hidden hook disabled by flag")
            model._hook_registered = False
        else:
            model._hook_registered = adapter.register_hidden_hook(
                backbone, model._last_hidden)
        if not model._hook_registered:
            logger.info(f"{model_type}: no hook — will use output_hidden_states=True")

        return model

    # ----------------------------------------------------------------
    # Prompt helpers (mirrors DreamRetriever)
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
    # Structural tail tokens (matches zero-shot dream/llada retrievers)
    # ----------------------------------------------------------------

    def _build_tail_ids(self) -> List[int]:
        """Build structural tail tokens: [quote, chat_end, EOS].

        Matches the zero-shot DreamRetriever._build_tail_ids and
        LLaDA2Retriever._build_tail_ids so that training sees the same
        sequence layout as zero-shot inference.
        """
        if self._n_tail <= 0:
            return []

        eos_id = self.tokenizer.eos_token_id
        tail: List[int] = []

        # Closing quote "
        quote_ids = self.tokenizer.encode('"', add_special_tokens=False)
        if len(quote_ids) == 1:
            tail.append(quote_ids[0])
        else:
            logger.warning('quote \'"\' tokenises to %d tokens; using EOS in slot 0', len(quote_ids))
            tail.append(eos_id)

        # Chat-template end token (im_end for Dream/Qwen, eot_id for LLaDA)
        _KNOWN_EOT = {'<|im_end|>': None, '<|eot_id|>': 126348}  # fallback for LLaDA
        for tok_str, fallback_id in _KNOWN_EOT.items():
            tid = self.tokenizer.convert_tokens_to_ids(tok_str)
            unk = getattr(self.tokenizer, 'unk_token_id', None)
            if tid is not None and tid >= 0 and tid != unk and tid != eos_id:
                tail.append(tid)
                break
            elif fallback_id is not None and fallback_id != eos_id:
                tail.append(fallback_id)
                break

        # Fill remaining slots with EOS
        while len(tail) < self._n_tail:
            tail.append(eos_id)
        return tail[:self._n_tail]

    # ----------------------------------------------------------------
    # Tokenization (called from the data collator, runs on CPU)
    # ----------------------------------------------------------------

    def tokenize(
        self, texts: List[str], is_query: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_ids, attention_mask) tensors on CPU."""
        prefix_ids = self._query_prefix_ids if is_query else self._passage_prefix_ids
        suffix_ids = self._query_suffix_ids if is_query else self._passage_suffix_ids
        gen_ids = [self.mask_token_id] * self._k(is_query)
        mask_block = gen_ids + self._tail_ids  # [MASK×K | " | chat_end | EOS]
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
            prefix_ids + ids + suffix_ids + mask_block
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
    # Attention mask: bidirectional (padding-only masking)
    # ----------------------------------------------------------------

    def _build_4d_mask(
        self, seq_len: int, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """4D [B, 1, S, S] bidirectional attention mask."""
        dtype = next(self.backbone.parameters()).dtype
        min_val = torch.finfo(dtype).min
        B = attention_mask.size(0)
        mask_4d = torch.zeros(B, 1, seq_len, seq_len,
                              device=attention_mask.device, dtype=dtype)
        pad = ~attention_mask.bool()
        mask_4d = mask_4d.masked_fill(pad.unsqueeze(1).unsqueeze(1), min_val)
        mask_4d = mask_4d.masked_fill(pad.unsqueeze(1).unsqueeze(3), min_val)
        return mask_4d

    # ----------------------------------------------------------------
    # Backbone forward
    # ----------------------------------------------------------------

    def _fwd(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        need_logits: bool = False,
        mask_4d: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Single backbone pass. Returns (hidden [B,L,H], logits [B,L,V] or None).

        mask_4d: pre-computed 4D attention mask — pass it in the multi-step loop
        to avoid recomputing the same mask at every denoising step.
        inputs_embeds: if provided, use these embeddings instead of input_ids.
        """
        # Adapter determines mask format: 4D [B,1,L,L] or 2D [B,L]
        use_4d = self.adapter.needs_4d_mask() if self.adapter else True
        if use_4d:
            if mask_4d is None:
                seq_len = inputs_embeds.size(1) if inputs_embeds is not None else input_ids.size(1)
                mask_4d = self._build_4d_mask(seq_len, attention_mask)
            fwd_mask = mask_4d
        else:
            fwd_mask = attention_mask

        # If hook is registered, skip output_hidden_states (saves ~1.2GB).
        need_hidden_states = not self._hook_registered
        fwd_kwargs = dict(
            attention_mask=fwd_mask,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        if inputs_embeds is not None:
            fwd_kwargs['inputs_embeds'] = inputs_embeds
        else:
            fwd_kwargs['input_ids'] = input_ids
        outputs = self.backbone(**fwd_kwargs)
        # Keep hidden in native dtype (bf16). Callers convert only small slices.
        if self._hook_registered and 'h' in self._last_hidden:
            hidden = self._last_hidden.pop('h')
        elif getattr(outputs, 'last_hidden_state', None) is not None:
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        else:
            hidden = outputs[0]

        # Keep logits in native dtype (bf16) to avoid doubling memory.
        # Downstream consumers convert small slices to float32 as needed.
        logits = outputs.logits if need_logits and hasattr(outputs, 'logits') else None

        if (self.debug_compare_hidden_once
                and self._hook_registered
                and not self._debug_hidden_compared):
            should_log = (not torch.distributed.is_initialized()
                          or torch.distributed.get_rank() == 0)
            if should_log:
                with torch.no_grad():
                    ref_outputs = self.backbone(
                        input_ids=input_ids,
                        attention_mask=fwd_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    if getattr(ref_outputs, 'last_hidden_state', None) is not None:
                        ref_hidden = ref_outputs.last_hidden_state
                    elif hasattr(ref_outputs, 'hidden_states') and ref_outputs.hidden_states is not None:
                        ref_hidden = ref_outputs.hidden_states[-1]
                    else:
                        ref_hidden = ref_outputs[0]
                    diff = (hidden.float() - ref_hidden.float()).abs()
                    logger.warning(
                        "Hidden-hook check: model=%s shape=%s mean_abs_diff=%.6g max_abs_diff=%.6g",
                        self.model_type, tuple(hidden.shape),
                        diff.mean().item(), diff.max().item(),
                    )
            self._debug_hidden_compared = True
        return hidden, logits

    # ----------------------------------------------------------------
    # Confidence-based unmasking
    # ----------------------------------------------------------------

    @staticmethod
    def _sample_with_confidence(logits: torch.Tensor, alg: str = 'entropy'):
        """Greedy decode + confidence score for discrete diffusion unmasking.

        Args:
            logits: [N, V] logits at the N currently-masked positions
            alg: 'entropy' (neg entropy; higher = more certain) or 'max_prob'
        Returns:
            (confidence [N], x0 [N]) — confidence scores and predicted token IDs
        """
        probs = torch.softmax(logits.float(), dim=-1)
        x0 = probs.argmax(dim=-1)
        if alg == 'entropy':
            log_probs = torch.log(probs + 1e-10)
            confidence = (probs * log_probs).sum(dim=-1)   # neg entropy
        else:
            confidence = probs.max(dim=-1).values
        return confidence, x0

    @torch.no_grad()
    def _unmask_step(
        self,
        curr_ids: torch.Tensor,    # [B, L]
        logits: torch.Tensor,      # [B, L, V]
        K: int,
        n_per_step: int,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """Unmask n_per_step most-confident gen tokens (uniform LLaDA-style schedule).

        Returns:
            new_ids: updated token IDs (same shape as curr_ids)
            newly_decoded: list-of-lists; newly_decoded[i] = gen-block positions
                           [0..K-1] that were just decoded for example i
        """
        new_ids = curr_ids.clone()
        B, L = curr_ids.shape
        g_start = L - K - self._n_tail
        newly_decoded: List[List[int]] = [[] for _ in range(B)]

        for i in range(B):
            gen_ids = new_ids[i, g_start:g_start + K]
            mask_bool = (gen_ids == self.mask_token_id)
            n_masked = mask_bool.sum().item()
            if n_masked == 0:
                continue
            # Convert only the small slice [n_masked, V] to float32, not full logits
            ml = logits[i, g_start:g_start + K][mask_bool].detach().float()
            if not torch.isfinite(ml).all():
                continue
            conf, x0 = self._sample_with_confidence(ml)
            n_tr = min(n_per_step, n_masked)
            _, xfer = torch.topk(conf, n_tr)
            masked_abs = torch.where(mask_bool)[0]
            selected = masked_abs[xfer]
            new_ids[i, g_start + selected] = x0[xfer]
            newly_decoded[i] = selected.tolist()

        return new_ids, newly_decoded

    # ----------------------------------------------------------------
    # Soft-token multi-step helpers
    # ----------------------------------------------------------------

    def _get_embed_layer(self) -> nn.Module:
        """Return the input embedding layer (works through PEFT wrapper)."""
        return self.backbone.get_input_embeddings()

    def _soft_unmask_step(
        self,
        curr_embeds: torch.Tensor,     # [B, L, H_emb] — current embeddings (differentiable)
        curr_ids: torch.Tensor,        # [B, L] — tracking which positions are still MASK
        logits: torch.Tensor,          # [B, L, V] — logits from current forward pass
        K: int,
        n_per_step: int,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Soft-unmask n_per_step most confident gen tokens.

        Instead of hard token replacement (argmax → embed lookup), computes
        differentiable soft embeddings: softmax(logits / τ) @ embedding_matrix.
        Gradient flows through logits → softmax → matmul → next-step embeddings.

        Returns:
            new_embeds: [B, L, H_emb] with soft tokens at decoded positions
            new_ids: [B, L] updated token IDs (hard argmax, for tracking only)
            newly_decoded: list of gen-block positions [0..K-1] decoded this step
        """
        embed_weight = self._get_embed_layer().weight  # [V, H_emb]
        B, L = curr_ids.shape
        g_start = L - K - self._n_tail
        new_ids = curr_ids.clone()
        newly_decoded: List[List[int]] = [[] for _ in range(B)]

        # Compute soft embeddings for all K gen positions (differentiable)
        gen_logits = logits[:, g_start:g_start + K, :]      # [B, K, V]
        soft_probs = F.softmax(gen_logits / temperature, dim=-1)  # [B, K, V]
        soft_embs = soft_probs @ embed_weight                # [B, K, H_emb]

        # Determine which positions to unmask (most confident, same schedule as hard)
        replace_mask = torch.zeros(B, K, dtype=torch.bool, device=curr_embeds.device)
        for i in range(B):
            gen_ids = curr_ids[i, g_start:g_start + K]
            mask_bool = (gen_ids == self.mask_token_id)
            n_masked = mask_bool.sum().item()
            if n_masked == 0:
                continue
            ml = gen_logits[i][mask_bool].detach().float()
            if not torch.isfinite(ml).all():
                continue
            conf, x0 = self._sample_with_confidence(ml)
            n_tr = min(n_per_step, n_masked)
            _, xfer = torch.topk(conf, n_tr)
            masked_abs = torch.where(mask_bool)[0]
            selected = masked_abs[xfer]
            replace_mask[i, selected] = True
            new_ids[i, g_start + selected] = x0[xfer]
            newly_decoded[i] = selected.tolist()

        # Mix: replace selected gen positions with soft embeddings, keep rest
        curr_gen_embs = curr_embeds[:, g_start:g_start + K, :]  # [B, K, H_emb]
        new_gen_embs = torch.where(
            replace_mask.unsqueeze(-1),   # [B, K, 1]
            soft_embs,                     # [B, K, H_emb] — differentiable
            curr_gen_embs,                 # [B, K, H_emb]
        )

        # Rebuild full sequence (avoids in-place ops on grad-carrying tensor)
        new_embeds = torch.cat([
            curr_embeds[:, :g_start, :],
            new_gen_embs,
            curr_embeds[:, g_start + K:, :],
        ], dim=1)

        return new_embeds, new_ids, newly_decoded

    def _soft_multistep_forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        passage_input_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
        query_content_ids: Optional[List],
        passage_content_ids: Optional[List],
        n_steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Multi-step denoising training with soft (differentiable) token replacement.

        Unlike hard multi-step where decoded positions are frozen (.detach()),
        soft-token mode keeps gradient flowing through ALL K positions at every
        step via softmax(logits/τ) @ embedding_matrix.

        Per-step contrastive loss (GIRCSE-style) provides direct supervision at
        each denoising step. A monotonicity regularizer penalizes regressions.
        """
        K_q, K_p = self.n_gen_q_tokens, self.n_gen_p_tokens
        n_tail = self._n_tail
        n_per_step_q = max(1, K_q // n_steps)
        n_per_step_p = max(1, K_p // n_steps)
        device = query_input_ids.device
        B_q = query_input_ids.size(0)
        B_p = passage_input_ids.size(0)
        n_paq = B_p // B_q
        L_q = query_input_ids.size(1)
        L_p = passage_input_ids.size(1)
        need_sparse = self.sparse_weight > 0

        # Pre-compute 4D masks (sequence structure doesn't change across steps)
        use_4d = self.adapter.needs_4d_mask() if self.adapter else True
        q_mask_4d = self._build_4d_mask(L_q, query_attention_mask) if use_4d else None
        p_mask_4d = self._build_4d_mask(L_p, passage_attention_mask) if use_4d else None

        # Initial embeddings from the backbone's embedding layer
        embed_layer = self._get_embed_layer()
        q_embeds = embed_layer(query_input_ids)
        p_embeds = embed_layer(passage_input_ids)
        q_curr_ids = query_input_ids.clone()
        p_curr_ids = passage_input_ids.clone()

        q_g = L_q - K_q - n_tail
        p_g = L_p - K_p - n_tail

        step_losses: List[torch.Tensor] = []
        final_loss: Dict[str, torch.Tensor] = {}

        for step in range(n_steps):
            is_last = (step == n_steps - 1)

            # Forward pass using embeddings (step 0: original, step 1+: soft tokens)
            q_h, q_logits = self._fwd(
                q_curr_ids, query_attention_mask,
                need_logits=True, mask_4d=q_mask_4d,
                inputs_embeds=q_embeds if step > 0 else None,
            )
            p_h, p_logits = self._fwd(
                p_curr_ids, passage_attention_mask,
                need_logits=True, mask_4d=p_mask_4d,
                inputs_embeds=p_embeds if step > 0 else None,
            )

            # Extract representations — ALL K positions have gradient
            q_repr_hidden = q_h[:, q_g:q_g + K_q, :].float()
            q_quotation_emb = q_h[:, q_g - 1, :].float()
            p_repr_hidden = p_h[:, p_g:p_g + K_p, :].float()
            p_quotation_emb = p_h[:, p_g - 1, :].float()

            q_sparse_max = None
            p_sparse_max = None
            # Monotonic trick: log1p(relu(.)) is non-decreasing, so
            #   max_k log1p(relu(x_k)) == log1p(relu(max_k x_k))
            # Maxing in bf16 first avoids materialising [B, K, V] fp32 intermediate.
            if need_sparse and q_logits is not None:
                q_sparse_max = torch.log1p(torch.relu(
                    q_logits[:, q_g:q_g + K_q, :].max(dim=1).values))
            if need_sparse and p_logits is not None:
                p_sparse_max = torch.log1p(torch.relu(
                    p_logits[:, p_g:p_g + K_p, :].max(dim=1).values))

            if self.normalize:
                q_quotation_emb = F.normalize(q_quotation_emb, p=2, dim=-1)
                q_repr_hidden = F.normalize(q_repr_hidden, p=2, dim=-1)
                p_quotation_emb = F.normalize(p_quotation_emb, p=2, dim=-1)
                p_repr_hidden = F.normalize(p_repr_hidden, p=2, dim=-1)

            q_repr = {'repr_hidden': q_repr_hidden, 'quotation_emb': q_quotation_emb}
            p_repr = {'repr_hidden': p_repr_hidden, 'quotation_emb': p_quotation_emb}
            if q_sparse_max is not None:
                if query_content_ids is not None:
                    q_sparse_max = filter_sparse(q_sparse_max, query_content_ids)
                q_repr['sparse_acts'] = q_sparse_max
            if p_sparse_max is not None:
                if passage_content_ids is not None:
                    p_sparse_max = filter_sparse(p_sparse_max, passage_content_ids)
                p_repr['sparse_acts'] = p_sparse_max

            # Cross-GPU negative sharing
            q_repr = self._gather_repr(q_repr)
            p_repr = self._gather_repr(p_repr)
            B_q_g = q_repr['repr_hidden'].size(0)
            B_p_g = p_repr['repr_hidden'].size(0)
            labels = torch.arange(B_q_g, device=device) * (B_p_g // B_q_g)

            # Per-step contrastive loss
            step_loss_dict = self.compute_loss(q_repr, p_repr, labels)
            step_losses.append(step_loss_dict['loss'])

            if is_last:
                final_loss = step_loss_dict

            # Soft unmask for next step (per-side K)
            if not is_last:
                q_embeds, q_curr_ids, _ = self._soft_unmask_step(
                    q_embeds, q_curr_ids, q_logits, K_q, n_per_step_q,
                    self.soft_temperature)
                p_embeds, p_curr_ids, _ = self._soft_unmask_step(
                    p_embeds, p_curr_ids, p_logits, K_p, n_per_step_p,
                    self.soft_temperature)

        # Accumulate per-step losses (linearly increasing weight)
        if len(step_losses) > 1:
            progressive_loss = sum(
                (s + 1) / n_steps * loss
                for s, loss in enumerate(step_losses[:-1])
            )
            final_loss['loss'] = final_loss['loss'] + progressive_loss
            final_loss['loss_progressive'] = progressive_loss.detach()

            # Monotonicity regularizer: penalize if later step is worse
            mono_penalty = torch.stack([
                torch.relu(step_losses[k + 1] - step_losses[k].detach())
                for k in range(len(step_losses) - 1)
            ]).mean()
            final_loss['loss'] = final_loss['loss'] + 0.1 * mono_penalty
            final_loss['loss_monotonicity'] = mono_penalty.detach()

        # DDP scaling
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            final_loss['loss'] = final_loss['loss'] * torch.distributed.get_world_size()

        # Denoising auxiliary (same as other paths)
        if self.denoising_weight > 0 and final_loss:
            p_corrupted, p_denoise_targets, mask_ratio = self._apply_text_masking(
                passage_input_ids, passage_attention_mask)
            _, p_logits_dn = self._fwd(p_corrupted, passage_attention_mask,
                                        need_logits=True, mask_4d=p_mask_4d)
            denoise_loss = self.compute_denoising_loss(
                p_logits_dn, p_denoise_targets, mask_ratio)
            final_loss['loss'] = final_loss['loss'] + self.denoising_weight * denoise_loss
            final_loss['loss_denoising'] = denoise_loss.detach()

        return final_loss

    # ----------------------------------------------------------------
    # K pre-encoder (two-stage variable-length training)
    # ----------------------------------------------------------------

    def _pre_encoder_logits(
        self,
        input_ids: torch.Tensor,           # [B, L]
        attention_mask: torch.Tensor,      # [B, L]
        is_query: bool,
    ) -> torch.Tensor:
        """Tiny embedding-only K-router.  Mean-pools the embedding-layer
        output over real (non-pad) tokens, then runs a small MLP head.
        Returns [B, n_K_pe] logits over self.k_pre_encoder_options.

        Cost ≈ embedding lookup + 2-layer MLP — under 1% of full forward.
        """
        head = self.k_pre_encoder_q if is_query else self.k_pre_encoder_p
        # Use main encoder's embedding layer (parameter-shared).
        embed_layer = self._get_embed_layer()
        emb = embed_layer(input_ids)  # [B, L, H]
        # Mean-pool over real tokens (attention_mask = 0 at pads).
        m = attention_mask.float().unsqueeze(-1)
        pooled = (emb.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # [B, H]
        # Match head dtype to avoid AMP/DeepSpeed dtype mismatch.
        head_dtype = next(head.parameters()).dtype
        logits = head(pooled.to(dtype=head_dtype)).float()  # [B, n_K_pe]
        return logits

    def _slice_to_per_item_K(
        self,
        input_ids: torch.Tensor,           # [B, L_max] left-padded, K_max MASKs at end
        attention_mask: torch.Tensor,
        K_per_item: torch.Tensor,          # [B] integers in self.k_pre_encoder_options
        K_max_side: int,                   # n_gen_q_tokens or n_gen_p_tokens for this side
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slice each item's input from K_max MASK tokens down to K MASK tokens
        (per-item K), preserving the structural tail [", chat_end, EOS].

        Layout (all left-padded — content right-aligned):
          [PAD ... PAD | prefix | text | suffix | K_max MASKs | tail]
        After slicing item with K_i:
          [PAD ... PAD | prefix | text | suffix |   K_i MASKs | tail]
        Batch is then padded to max-K-in-batch length.
        """
        B = input_ids.size(0)
        L_in = input_ids.size(1)
        n_tail = self._n_tail
        K_per_item = K_per_item.tolist() if torch.is_tensor(K_per_item) else list(K_per_item)
        # Per-item sliced sequences.
        sliced_ids: List[torch.Tensor] = []
        sliced_mask: List[torch.Tensor] = []
        for i in range(B):
            K_i = int(K_per_item[i])
            n_remove = K_max_side - K_i
            # Cut tokens [-(n_tail + n_remove) : -n_tail).
            # Equivalent: keep input_ids[: L_in - n_tail - n_remove] + input_ids[-n_tail:]
            keep_until = L_in - n_tail - n_remove
            new_ids = torch.cat([input_ids[i, :keep_until], input_ids[i, -n_tail:]
                                 if n_tail > 0 else input_ids.new_empty(0)])
            new_mask = torch.cat([attention_mask[i, :keep_until], attention_mask[i, -n_tail:]
                                  if n_tail > 0 else attention_mask.new_empty(0)])
            sliced_ids.append(new_ids)
            sliced_mask.append(new_mask)

        # Pad LEFT to max length so the structural tail stays at position -1.
        L_out = max(s.size(0) for s in sliced_ids)
        pad_id = (self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)
        out_ids = input_ids.new_full((B, L_out), pad_id)
        out_mask = attention_mask.new_zeros((B, L_out))
        for i, (s_ids, s_mask) in enumerate(zip(sliced_ids, sliced_mask)):
            L_i = s_ids.size(0)
            out_ids[i, L_out - L_i:] = s_ids
            out_mask[i, L_out - L_i:] = s_mask
        return out_ids, out_mask

    def _extract_repr_per_K(
        self,
        hidden: torch.Tensor,              # [B, L_out, H]
        K_per_item: torch.Tensor,          # [B] int K per item
        n_K_max: int,                      # max K across self.k_pre_encoder_options
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract per-item repr_hidden using each item's K.
        Returns:
          repr_hidden: [B, n_K_max, H] — first K_i positions filled, rest zero
          quotation_emb: [B, H] — position before first MASK per item
        With left-padding, the rightmost positions per item are:
          [..., gen_K_i MASKs, tail tokens (n_tail)]
        So mask block = positions [L_out - n_tail - K_i, L_out - n_tail).
        Quotation = position L_out - n_tail - K_i - 1.
        """
        B, L_out, H = hidden.shape
        n_tail = self._n_tail
        K_per_item = K_per_item.tolist() if torch.is_tensor(K_per_item) else list(K_per_item)
        repr_hidden = hidden.new_zeros(B, n_K_max, H)
        quotation_emb = hidden.new_zeros(B, H)
        for i in range(B):
            K_i = int(K_per_item[i])
            mask_end = L_out - n_tail
            mask_start = mask_end - K_i
            repr_hidden[i, :K_i] = hidden[i, mask_start:mask_end].float()
            if mask_start - 1 >= 0:
                quotation_emb[i] = hidden[i, mask_start - 1].float()
        return repr_hidden, quotation_emb

    def _gumbel_st_sample_K(
        self,
        logits: torch.Tensor,              # [B, n_K_pe]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gumbel-softmax sample with straight-through.

        Returns:
          K_per_item: [B] int — actual K integer values to use for slicing
          soft_one_hot: [B, n_K_pe] — hard one-hot in fwd, soft in backward
                                       (used for cost regularizer + ST credit)
          chosen_prob: [B] — one-hot's value at chosen index (always 1.0 in fwd)
                              used to inject gradient back through ST
        """
        soft_one_hot = F.gumbel_softmax(
            logits, tau=max(self.gumbel_temperature, 1e-3), hard=True
        )  # [B, n_K_pe]
        # Integer K from one-hot
        K_idx = soft_one_hot.argmax(dim=-1)  # [B]
        K_per_item = self.k_pe_options_tensor[K_idx].long()  # [B]
        # ST credit: chosen_prob is always 1.0 in forward, has gradient via ST
        chosen_prob = soft_one_hot.gather(-1, K_idx.unsqueeze(-1)).squeeze(-1)
        return K_per_item, soft_one_hot, chosen_prob

    def _forward_with_k_pre_encoder(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        passage_input_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
        query_content_ids: Optional[List],
        passage_content_ids: Optional[List],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Two-stage forward: pre-encoder predicts K_q, K_p per item; main
        encoder runs at variable length (per-item K).  Real encoding savings
        when batch's max-K is below K_max.

        Backprop:
          - Main encoder: gradient flows through standard InfoNCE on per-item
            sliced representations.
          - Pre-encoder: gradient flows via Gumbel straight-through on
            chosen_prob (always 1.0 in fwd; gradient injected through ST).
            Plus a cost regularizer λ * E[K] discourages collapsing to K=K_max.
        """
        device = labels.device
        K_max_q = self.n_gen_q_tokens
        K_max_p = self.n_gen_p_tokens

        # ── 1. Pre-encoder forward (cheap — embedding lookup + small MLP).
        q_logits = self._pre_encoder_logits(
            query_input_ids, query_attention_mask, is_query=True)
        p_logits = self._pre_encoder_logits(
            passage_input_ids, passage_attention_mask, is_query=False)

        # ── 2. Gumbel-ST sample K per item (queries + passages).
        K_q_per_item, q_soft, q_chosen = self._gumbel_st_sample_K(q_logits)
        K_p_per_item, p_soft, p_chosen = self._gumbel_st_sample_K(p_logits)

        # Clamp at the per-side K_max in case the pre-encoder options exceed it.
        # (Shouldn't happen by construction but a safety net.)
        K_q_per_item = K_q_per_item.clamp(min=1, max=K_max_q)
        K_p_per_item = K_p_per_item.clamp(min=1, max=K_max_p)

        # ── 3. Slice each item's input to its K (variable-length batch).
        q_ids_sliced, q_mask_sliced = self._slice_to_per_item_K(
            query_input_ids, query_attention_mask, K_q_per_item, K_max_q)
        p_ids_sliced, p_mask_sliced = self._slice_to_per_item_K(
            passage_input_ids, passage_attention_mask, K_p_per_item, K_max_p)

        # ── 4. Main encoder forward (one pass each at variable length).
        need_sparse = self.sparse_weight > 0
        q_hidden, q_logits_main = self._fwd(
            q_ids_sliced, q_mask_sliced, need_logits=need_sparse)
        p_hidden, p_logits_main = self._fwd(
            p_ids_sliced, p_mask_sliced, need_logits=need_sparse)

        # ── 5. Per-item repr extraction.
        n_K_max = max(self.k_pre_encoder_options)
        q_repr_hidden, q_quotation_emb = self._extract_repr_per_K(
            q_hidden, K_q_per_item, n_K_max)
        p_repr_hidden, p_quotation_emb = self._extract_repr_per_K(
            p_hidden, K_p_per_item, n_K_max)

        # ── 6. Sparse: max-pool over each item's K positions.
        q_sparse_max = None
        p_sparse_max = None
        if need_sparse and q_logits_main is not None:
            # For each item, compute log(1+relu(...)) over its K MASK positions,
            # then max-pool across positions.  Same as legacy sparse computation.
            B_q = q_ids_sliced.size(0)
            L_q_out = q_ids_sliced.size(1)
            q_sparse_max = q_logits_main.new_zeros(B_q, q_logits_main.size(-1))
            for i in range(B_q):
                K_i = int(K_q_per_item[i].item())
                mask_end = L_q_out - self._n_tail
                mask_start = mask_end - K_i
                slogits = q_logits_main[i, mask_start:mask_end]
                # Monotonic trick: max over K positions first → log1p(relu()) on [V].
                q_sparse_max[i] = torch.log1p(torch.relu(slogits.max(dim=0).values))
        if need_sparse and p_logits_main is not None:
            B_p = p_ids_sliced.size(0)
            L_p_out = p_ids_sliced.size(1)
            p_sparse_max = p_logits_main.new_zeros(B_p, p_logits_main.size(-1))
            for i in range(B_p):
                K_i = int(K_p_per_item[i].item())
                mask_end = L_p_out - self._n_tail
                mask_start = mask_end - K_i
                slogits = p_logits_main[i, mask_start:mask_end]
                # Monotonic trick: max over K positions first → log1p(relu()) on [V].
                p_sparse_max[i] = torch.log1p(torch.relu(slogits.max(dim=0).values))

        # ── 7. Normalize.
        if self.normalize:
            q_repr_hidden = F.normalize(q_repr_hidden, p=2, dim=-1)
            q_quotation_emb = F.normalize(q_quotation_emb, p=2, dim=-1)
            p_repr_hidden = F.normalize(p_repr_hidden, p=2, dim=-1)
            p_quotation_emb = F.normalize(p_quotation_emb, p=2, dim=-1)

        # ── 8. ST credit injection — multiply repr by chosen_prob (always 1.0
        # in fwd, has gradient through ST in backward).  This is the standard
        # Gumbel-ST trick for credit assignment when downstream operations
        # (slicing, reshape) are non-differentiable.
        q_repr_hidden = q_repr_hidden * q_chosen.float().unsqueeze(-1).unsqueeze(-1)
        p_repr_hidden = p_repr_hidden * p_chosen.float().unsqueeze(-1).unsqueeze(-1)
        q_quotation_emb = q_quotation_emb * q_chosen.float().unsqueeze(-1)
        p_quotation_emb = p_quotation_emb * p_chosen.float().unsqueeze(-1)

        # Build repr dicts and apply content filter to sparse.
        q_repr = {'repr_hidden': q_repr_hidden, 'quotation_emb': q_quotation_emb}
        p_repr = {'repr_hidden': p_repr_hidden, 'quotation_emb': p_quotation_emb}
        if q_sparse_max is not None:
            if query_content_ids is not None:
                q_sparse_max = filter_sparse(q_sparse_max, query_content_ids)
            q_repr['sparse_acts'] = q_sparse_max
        if p_sparse_max is not None:
            if passage_content_ids is not None:
                p_sparse_max = filter_sparse(p_sparse_max, passage_content_ids)
            p_repr['sparse_acts'] = p_sparse_max

        # ── 9. Cross-GPU gather + recompute labels for gathered batch.
        q_repr = self._gather_repr(q_repr)
        p_repr = self._gather_repr(p_repr)
        B_q_all = q_repr['repr_hidden'].size(0)
        B_p_all = p_repr['repr_hidden'].size(0)
        n_paq_g = B_p_all // B_q_all
        labels_g = torch.arange(B_q_all, device=device) * n_paq_g

        # ── 10. Standard InfoNCE on per-item-K representations (zero-padded
        # positions contribute nothing thanks to MaxSim's max-over-positions).
        loss_dict = self.compute_loss(q_repr, p_repr, labels_g)

        # ── 11. Cost regularizer — λ * E[K_q + K_p] using soft probs (so it's
        # differentiable w.r.t. logits).  Discourages collapsing to K_max.
        if self.k_cost_lambda > 0:
            K_options_t = self.k_pe_options_tensor  # [n_K_pe]
            expected_K_q = (q_soft * K_options_t).sum(dim=-1).mean()
            expected_K_p = (p_soft * K_options_t).sum(dim=-1).mean()
            cost_loss = (expected_K_q + expected_K_p) / float(max(K_options_t.max().item(), 1.0))
            loss_dict['loss'] = loss_dict['loss'] + self.k_cost_lambda * cost_loss
            loss_dict['k_cost'] = cost_loss.detach()
            loss_dict['expected_K_q'] = expected_K_q.detach()
            loss_dict['expected_K_p'] = expected_K_p.detach()

        # Diagnostics: marginal K distributions.
        with torch.no_grad():
            for i, K in enumerate(self.k_pre_encoder_options):
                loss_dict[f'pe_kq_p{K}'] = q_soft[:, i].mean().detach()
                loss_dict[f'pe_kp_p{K}'] = p_soft[:, i].mean().detach()

        # DDP loss scaling (counters HF Trainer's gradient averaging).
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            loss_dict['loss'] = loss_dict['loss'] * torch.distributed.get_world_size()

        return loss_dict

    # ----------------------------------------------------------------
    # Encode — single-pass or multi-step denoising (inference)
    # ----------------------------------------------------------------

    def encode(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        is_query: bool = False,
        compute_sparse: Optional[bool] = None,
        content_token_ids: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        PromptReps encoding.

        num_denoise_steps == 1: single forward pass (fast path, differentiable,
          used in both training and inference).
        num_denoise_steps >  1: iterative denoising loop (inference).
          repr_hidden[i, k] is captured at the step when gen token k transitions
          MASK → decoded, benefiting from previously-decoded token context.
          Uniform schedule: n_per_step = K // n_steps tokens decoded per step.
          quotation_emb is captured at the final step.
          Call under torch.no_grad() for inference; use forward() for training.

        Returns:
            repr_hidden   [B, K, H]   ColBERT repr (one vector per gen token)
            quotation_emb [B, H]      closing " token (single-dense repr)
            sparse_acts   [B, V]      sum of log(1+relu(logit)) across K positions
        """
        K = self._k(is_query)
        B = input_ids.size(0)
        L = input_ids.size(1)
        n_steps = self.num_denoise_steps
        device = input_ids.device
        g_start = L - K - self._n_tail  # left-padded: MASK block then structural tail

        if compute_sparse is None:
            need_sparse = self.sparse_weight > 0
        else:
            need_sparse = compute_sparse

        # Initialised here so both the single- and multi-step branches always leave
        # these bound — the multi-step path only ever populates sparse_max.
        sparse_max: Optional[torch.Tensor] = None
        sparse_per_pos: Optional[torch.Tensor] = None

        # ── Single-pass fast path (n_steps ≤ 1) ──────────────────────────────
        if n_steps <= 1:
            hidden, logits = self._fwd(input_ids, attention_mask, need_logits=need_sparse)
            repr_hidden = hidden[:, g_start:g_start + K, :].float()   # [B, K, H]
            quotation_emb = hidden[:, g_start - 1, :].float()         # [B, H]
            if need_sparse and logits is not None:
                # Per-position sparse acts for ColBERT-style multi-vector sparse
                # (caller may pop result['sparse_acts_per_pos'] to do per-K topk).
                # Keep in bf16 — the downstream topk → ×100 → int round path
                # doesn't need fp32 precision, and the fp32 cast was the
                # dominant memory bandwidth cost at K=16 (312 MB → 156 MB/batch
                # for V=152k, B=32; 2× reduction).
                gen_logits = logits[:, g_start:g_start + K, :]                  # [B, K, V] bf16
                sparse_per_pos = torch.log1p(torch.relu(gen_logits))            # [B, K, V] bf16
                # sparse_max stays fp32 (small, [B, V]) for downstream consumers.
                sparse_max = sparse_per_pos.max(dim=1).values.float()           # [B, V]    fp32

        # ── Multi-step denoising loop (n_steps > 1, inference) ───────────────
        else:
            curr_ids = input_ids.clone()
            n_per_step = max(1, K // n_steps)
            vocab_size = self.backbone.config.vocab_size

            repr_buf = torch.zeros(B, K, self.hidden_size, device=device)
            repr_saved = torch.zeros(B, K, dtype=torch.bool, device=device)
            sparse_max = (torch.zeros(B, vocab_size, device=device)
                          if need_sparse else None)
            quotation_emb = None
            mask_4d = self._build_4d_mask(L, attention_mask) if (self.adapter.needs_4d_mask() if self.adapter else True) else None

            for step in range(n_steps):
                hidden, logits = self._fwd(curr_ids, attention_mask, need_logits=True, mask_4d=mask_4d)
                is_last = (step == n_steps - 1)

                # ── Vectorized over batch (replaces `for i in range(B): ... .item() ...`).
                # Behavior preserved: greedy argmax, neg-entropy confidence, top-n_per_step
                # transitions per item; on last step all remaining masks transition.
                gen_ids_all   = curr_ids[:, g_start:g_start + K]                  # [B, K] int
                mask_bool_all = (gen_ids_all == self.mask_token_id)               # [B, K] bool
                ml_all        = logits[:, g_start:g_start + K, :]                 # [B, K, V] bf16

                # Confidence + argmax (matches _sample_with_confidence(alg='entropy'))
                probs     = F.softmax(ml_all.float(), dim=-1)                     # [B, K, V] fp32
                log_probs = torch.log(probs + 1e-10)
                conf      = (probs * log_probs).sum(dim=-1)                       # [B, K] (neg entropy)
                x0_pred   = probs.argmax(dim=-1)                                  # [B, K]
                # Restrict topk to currently-masked positions
                conf = torch.where(mask_bool_all, conf, torch.full_like(conf, -float('inf')))

                if is_last:
                    xfer_mask = mask_bool_all                                     # all remaining masks
                else:
                    n_tr = min(n_per_step, K)
                    _, top_idx = conf.topk(n_tr, dim=-1)                          # [B, n_tr]
                    xfer_mask = torch.zeros_like(mask_bool_all)
                    xfer_mask.scatter_(1, top_idx, True)
                    xfer_mask = xfer_mask & mask_bool_all                         # only actually-masked positions

                # ── Vectorized state updates ────────────────────────────────────
                # 1. curr_ids: replace MASK with predicted at xfer positions
                if xfer_mask.any():
                    new_K = torch.where(xfer_mask, x0_pred, gen_ids_all)
                    # In-place via clone (curr_ids may be a view of input_ids)
                    curr_ids = curr_ids.clone()
                    curr_ids[:, g_start:g_start + K] = new_K

                # 2. repr_buf: save hidden at to_save positions (xfer & not yet saved)
                to_save = xfer_mask & ~repr_saved
                if to_save.any():
                    hidden_chunk = hidden[:, g_start:g_start + K, :]              # [B, K, H]
                    repr_buf = torch.where(to_save.unsqueeze(-1), hidden_chunk, repr_buf)
                    # 3. sparse_max: max over to_save positions of log(1+relu(logits))
                    if sparse_max is not None:
                        sparse_chunk = torch.log1p(torch.relu(ml_all))            # [B, K, V] bf16
                        masked_sparse = torch.where(
                            to_save.unsqueeze(-1).expand_as(sparse_chunk),
                            sparse_chunk,
                            torch.full_like(sparse_chunk, -float('inf')))
                        new_max = masked_sparse.max(dim=1).values.float()         # [B, V] fp32
                        sparse_max = torch.max(sparse_max, new_max)
                    repr_saved = repr_saved | to_save

                if is_last:
                    quotation_emb = hidden[:, g_start - 1, :].float()
                    # Catch positions that were never masked (rare but possible
                    # if input already had a non-MASK token in the K block).
                    unsaved = ~repr_saved
                    if unsaved.any():
                        hidden_chunk = hidden[:, g_start:g_start + K, :]
                        repr_buf = torch.where(unsaved.unsqueeze(-1), hidden_chunk, repr_buf)
                        if sparse_max is not None:
                            sparse_chunk = torch.log1p(torch.relu(ml_all))
                            masked_sparse = torch.where(
                                unsaved.unsqueeze(-1).expand_as(sparse_chunk),
                                sparse_chunk,
                                torch.full_like(sparse_chunk, -float('inf')))
                            new_max = masked_sparse.max(dim=1).values.float()
                            sparse_max = torch.max(sparse_max, new_max)
                        repr_saved = repr_saved | unsaved

            repr_buf = torch.nan_to_num(repr_buf)
            repr_hidden = repr_buf
            quotation_emb = torch.nan_to_num(quotation_emb)

        # ── Normalize ─────────────────────────────────────────────────────────
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
        # Per-position sparse [B, K, V] — single-pass only, K>1
        if sparse_per_pos is not None and K > 1:
            result['sparse_acts_per_pos'] = sparse_per_pos  # unfiltered; caller applies content filter
        return result

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
        # Replace own shard with original (keeps gradient)
        gathered[torch.distributed.get_rank()] = t
        return torch.cat(gathered, dim=0)

    def _gather_repr(self, repr_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Gather all representation tensors across GPUs."""
        out = {}
        for k, v in repr_dict.items():
            out[k] = self._dist_gather(v)
        return out

    # ----------------------------------------------------------------
    # MaxSim (ColBERT-style)
    # ----------------------------------------------------------------

    @staticmethod
    def maxsim(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        q: [B_q, k, H], p: [B_p, k, H]  →  [B_q, B_p] MaxSim scores.
        """
        sims = torch.einsum('ikh,jlh->ijkl', q, p)       # [B_q, B_p, k_q, k_p]
        return sims.max(dim=-1).values.sum(dim=-1)        # [B_q, B_p]

    # ----------------------------------------------------------------
    # Auxiliary losses (diffusion-native)
    # ----------------------------------------------------------------

    @torch.no_grad()
    def _corrupt_text(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
        rate: float,
    ) -> torch.Tensor:
        """Randomly replace rate% of text tokens with MASK for corruption augmentation.

        Only corrupts real text tokens (not padding, not gen MASKs, not EOS).
        Returns a new tensor (does not modify input in-place).
        """
        B, L = input_ids.shape
        K = self.n_gen_p_tokens                      # only called on passages
        g_start = L - K - self._n_tail

        corrupted = input_ids.clone()
        is_real = attention_mask.bool()
        is_gen = torch.zeros(L, dtype=torch.bool, device=input_ids.device)
        is_gen[g_start:] = True
        is_candidate = is_real & ~is_gen.unsqueeze(0) & (input_ids != self.mask_token_id)

        for i in range(B):
            cand_idx = torch.where(is_candidate[i])[0]
            if cand_idx.numel() == 0:
                continue
            n_mask = max(1, int(cand_idx.numel() * rate))
            perm = torch.randperm(cand_idx.numel(), device=input_ids.device)[:n_mask]
            corrupted[i, cand_idx[perm]] = self.mask_token_id

        return corrupted

    def _apply_text_masking(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Randomly mask text tokens (NOT generation MASKs) for denoising auxiliary loss.

        Returns:
            corrupted_ids: input_ids with some text tokens replaced by MASK
            denoise_targets: original token IDs at masked positions (-100 elsewhere)
            mask_ratio: fraction of tokens masked (t for 1/t weighting)
        """
        B, L = input_ids.shape
        K = self.n_gen_p_tokens                       # called on passages
        g_start = L - K - self._n_tail  # left-padded: gen block + structural tail

        corrupted = input_ids.clone()
        targets = torch.full_like(input_ids, -100)  # -100 = ignore in CE

        # Candidate text positions: not padding, not in gen/EOS block, not already MASK
        is_real = attention_mask.bool()                             # [B, L]
        is_gen = torch.zeros(L, dtype=torch.bool, device=input_ids.device)
        is_gen[g_start:] = True  # marks MASK positions + EOS as non-candidate
        is_candidate = is_real & ~is_gen.unsqueeze(0) & (input_ids != self.mask_token_id)

        for i in range(B):
            cand_idx = torch.where(is_candidate[i])[0]
            if cand_idx.numel() == 0:
                continue
            n_mask = max(1, int(cand_idx.numel() * self.denoise_mask_ratio))
            perm = torch.randperm(cand_idx.numel(), device=input_ids.device)[:n_mask]
            mask_positions = cand_idx[perm]
            targets[i, mask_positions] = input_ids[i, mask_positions]
            corrupted[i, mask_positions] = self.mask_token_id

        mask_ratio = self.denoise_mask_ratio
        return corrupted, targets, mask_ratio

    def compute_denoising_loss(
        self,
        logits: torch.Tensor,      # [B, L, V] from forward pass on corrupted input
        targets: torch.Tensor,      # [B, L] with -100 for non-masked positions
        mask_ratio: float,          # t for 1/t weighting
    ) -> torch.Tensor:
        """Compute Dream/LLaDA-style denoising loss: weighted CE on masked text tokens."""
        # Flatten and compute CE (ignores -100 positions automatically)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        # Weight by 1/t (matching Dream/LLaDA pre-training ELBO)
        return loss / max(mask_ratio, 1e-3)

    def compute_diversity_loss(
        self,
        repr_hidden: torch.Tensor,  # [B, K, H]
    ) -> torch.Tensor:
        """Push K representation vectors to be diverse (low pairwise cosine similarity)."""
        K = repr_hidden.size(1)
        if K <= 1:
            return torch.tensor(0.0, device=repr_hidden.device)

        # Pairwise cosine similarity [B, K, K] — repr_hidden is already L2-normalised
        sim_matrix = torch.bmm(repr_hidden, repr_hidden.transpose(1, 2))  # [B, K, K]

        # Mean of upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(K, K, device=repr_hidden.device), diagonal=1).bool()
        pairwise_sims = sim_matrix[:, mask]  # [B, K*(K-1)/2]

        # Hinge: penalize similarities above 0 (push toward orthogonal)
        diversity_loss = torch.relu(pairwise_sims).mean()
        return diversity_loss

    @staticmethod
    def _mean_offdiag_cos(repr_hidden: torch.Tensor) -> torch.Tensor:
        """Mean off-diagonal cosine similarity over K vectors."""
        K = repr_hidden.size(1)
        if K <= 1:
            return torch.tensor(0.0, device=repr_hidden.device)
        normed = F.normalize(repr_hidden.float(), p=2, dim=-1)
        sim = torch.bmm(normed, normed.transpose(1, 2))
        mask = torch.triu(torch.ones(K, K, device=repr_hidden.device), diagonal=1).bool()
        return sim[:, mask].mean()

    def _dense_debug_stats(
        self,
        q_repr: Dict[str, torch.Tensor],
        p_repr: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        q_raw: Optional[torch.Tensor] = None,
        p_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Summaries that explain whether dense loss comes from poor score gaps or collapsed vectors."""
        scores = self.maxsim(q_repr['repr_hidden'], p_repr['repr_hidden'])
        row_idx = torch.arange(scores.size(0), device=scores.device)
        pos = scores[row_idx, labels]
        neg = scores.clone()
        neg[row_idx, labels] = float('-inf')
        hard_neg = neg.max(dim=1).values

        stats = {
            'debug_pos_score': pos.mean().detach(),
            'debug_hardneg_score': hard_neg.mean().detach(),
            'debug_score_gap': (pos - hard_neg).mean().detach(),
            'debug_pos_beats_hardneg': (pos > hard_neg).float().mean().detach(),
            'debug_q_mask_cos': self._mean_offdiag_cos(q_repr['repr_hidden']).detach(),
            'debug_p_mask_cos': self._mean_offdiag_cos(p_repr['repr_hidden']).detach(),
        }

        if q_raw is not None:
            stats['debug_q_mask_norm'] = q_raw.float().norm(dim=-1).mean().detach()
            stats['debug_q_mask_raw_cos'] = self._mean_offdiag_cos(q_raw).detach()
        if p_raw is not None:
            stats['debug_p_mask_norm'] = p_raw.float().norm(dim=-1).mean().detach()
            stats['debug_p_mask_raw_cos'] = self._mean_offdiag_cos(p_raw).detach()

        return stats

    # ----------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------

    def compute_loss(
        self,
        q_repr: Dict[str, torch.Tensor],
        p_repr: Dict[str, torch.Tensor],
        labels: torch.Tensor,             # [B_q] index of positive in B_p
    ) -> Dict[str, torch.Tensor]:
        """
        Dense loss: ColBERT MaxSim on repr_hidden (MASK positions).
        For K=1 this reduces to dot product of the single MASK vector.
        Sparse InfoNCE (if sparse_weight > 0): raw dot product + temperature.

        K-adapter: when self.use_k_adapter is True, dispatch to the per-cell
        loss path that trains both the retriever and the K-routing head.
        """
        if self.use_k_adapter:
            return self._compute_loss_with_k_adapter(q_repr, p_repr, labels)

        device = labels.device
        K = q_repr['repr_hidden'].size(1)

        total_loss = torch.tensor(0.0, device=device)

        # Primary dense: ColBERT MaxSim on repr_hidden (MASK positions).
        colbert_scores = self.maxsim(q_repr['repr_hidden'], p_repr['repr_hidden'])
        colbert_loss = F.cross_entropy(colbert_scores / K / self.temperature, labels)
        total_loss = self.dense_weight * colbert_loss

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

        # Auxiliary: diversity loss on repr_hidden (if enabled)
        diversity_loss = torch.tensor(0.0, device=device)
        if self.diversity_weight > 0 and K > 1:
            # Average over query and passage batches (both need diverse vectors)
            diversity_loss = 0.5 * (
                self.compute_diversity_loss(q_repr['repr_hidden']) +
                self.compute_diversity_loss(p_repr['repr_hidden'])
            )
            total_loss = total_loss + self.diversity_weight * diversity_loss

        result = {
            'loss': total_loss,
            'loss_dense': (self.dense_weight * colbert_loss).detach(),
            'loss_sparse': sparse_loss.detach(),
        }
        if self.diversity_weight > 0:
            result['loss_diversity'] = diversity_loss.detach()
        # loss_denoising is added in forward() since it needs logits
        return result

    # ----------------------------------------------------------------
    # K-adapter loss
    # ----------------------------------------------------------------

    def _compute_loss_with_k_adapter(
        self,
        q_repr: Dict[str, torch.Tensor],
        p_repr: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Joint training of retriever + K-routing head.

        For each (K_q, K_p) cell in `self.k_adapter_options × self.k_adapter_options`:
          1. Slice prefix of q_repr / p_repr at that K.
          2. Compute MaxSim → fuse with sparse → CE loss vs in-batch negatives.
          3. Per-query loss tensor of shape [B_q, n_K_q, n_K_p].

        Teacher distribution per query: softmax(-per_cell_loss / τ_T).
        Adapter (small MLP on quotation_emb) predicts a softmax over cells.

        Adapter loss = KL(teacher || adapter_probs) — the adapter learns to
        mimic which cell the model itself prefers (lowest CE) without any
        oracle/MRR involvement.

        Retriever loss = sum_c adapter_probs[i, c] * per_cell_loss[i, c]:
          - early in training (uniform adapter): ~ matryoshka-style avg over
            all cells, every prefix gets gradient → all K positions trained.
          - late in training (peaky adapter): gradient concentrates on
            adapter-preferred cell → cell specialization.
        """
        device = labels.device
        K_opts = self.k_adapter_options
        n_K = self.n_K
        K_max_q = q_repr['repr_hidden'].size(1)
        K_max_p = p_repr['repr_hidden'].size(1)
        # If model was instantiated at smaller K than max(k_adapter_options),
        # filter the option list down to what we actually have positions for.
        valid_q = [k for k in K_opts if k <= K_max_q]
        valid_p = [k for k in K_opts if k <= K_max_p]
        # Index lookup back into the n_K-wide cell grid (so options outside
        # the model's K-range get teacher prob 0 / adapter logits ignored).
        idx_q = [K_opts.index(k) for k in valid_q]
        idx_p = [K_opts.index(k) for k in valid_p]

        # ── 1. Sparse score (K-invariant — added to every cell uniformly).
        # Stays max-pooled so it doesn't depend on K choice.
        sparse_scores = None
        if (self.sparse_weight > 0
                and 'sparse_acts' in q_repr
                and 'sparse_acts' in p_repr):
            sparse_scores = q_repr['sparse_acts'] @ p_repr['sparse_acts'].T
            # Note: sparse uses raw dot product (no temperature), matching the
            # legacy compute_loss path.  Adding it BEFORE softmax in CE won't
            # change ranking since it shifts every doc by the same per-query
            # amount — but it DOES affect per-cell CE values (different cells
            # have different dense magnitudes; sparse adds a constant).  This
            # is intentional: the per-cell loss reflects the actual fused-score
            # contrast the model is trained on.
            #
            # We deliberately apply temperature scaling to the dense part only,
            # mirroring how inference fuses dense (MaxSim/K) + sparse_weight*sparse.
        B_q = q_repr['repr_hidden'].size(0)

        # ── 2. Per-cell dense scores → per-cell CE losses.
        #     [B_q, B_p, K_max_q, K_max_p] is the full per-position dot product.
        #     We compute MaxSim per (K_q, K_p) by slicing prefixes.
        per_cell_losses = q_repr['repr_hidden'].new_zeros(B_q, n_K, n_K)
        # Precompute the "all positions" inner-product tensor once:
        #   sims_full[b_q, b_p, i, j] = q_repr[b_q, i] · p_repr[b_q_p_pair, j]
        # then maxsim(K_q, K_p) = sims_full[:, :, :K_q, :K_p].max(-1).sum(-1)
        sims_full = torch.einsum('bih,djh->bdij',
                                  q_repr['repr_hidden'].float(),
                                  p_repr['repr_hidden'].float())  # [B_q, B_p, K_max_q, K_max_p]

        for ii, K_q in enumerate(valid_q):
            ki = idx_q[ii]
            for jj, K_p in enumerate(valid_p):
                kj = idx_p[jj]
                # Slice → max over passage positions → sum over query positions.
                cell_dense = sims_full[:, :, :K_q, :K_p].max(dim=-1).values.sum(dim=-1)
                cell_dense = cell_dense / K_q  # MaxSim normalization, matches legacy
                cell_score = self.dense_weight * cell_dense / self.temperature
                if sparse_scores is not None:
                    cell_score = cell_score + self.sparse_weight * sparse_scores
                # Per-query CE — reduction='none' for [B_q] tensor, no batch mean.
                cell_loss = F.cross_entropy(cell_score, labels, reduction='none')  # [B_q]
                per_cell_losses[:, ki, kj] = cell_loss

        # ── 3. Teacher distribution: softmax(-loss/τ_T) over cells.
        #     Lower loss (better separation) → higher teacher weight.
        flat_losses = per_cell_losses.view(B_q, n_K * n_K)        # [B_q, n_K^2]
        teacher_probs = F.softmax(
            -flat_losses.detach() / max(self.teacher_temperature, 1e-6), dim=-1
        )

        # ── 4. Adapter prediction.  Input: query's quotation_emb (K-invariant
        #     content-aware summary).  Output: distribution over n_K^2 cells.
        if 'quotation_emb' not in q_repr:
            raise RuntimeError(
                "K-adapter requires 'quotation_emb' in q_repr.  "
                "encode() must populate it before compute_loss()."
            )
        # Match adapter weight dtype (DeepSpeed/AMP may cast adapter to bf16).
        # quotation_emb is fp32 from encode(); cast it to whatever the linear
        # weight is, run the MLP, then cast logits back to fp32 for stable
        # log_softmax + KL arithmetic.
        adapter_dtype = next(self.k_adapter.parameters()).dtype
        adapter_input = q_repr['quotation_emb'].to(dtype=adapter_dtype)  # [B_q, H]
        adapter_logits = self.k_adapter(adapter_input).float()           # [B_q, n_K^2]
        adapter_log_probs = F.log_softmax(adapter_logits, dim=-1)
        adapter_probs = adapter_log_probs.exp()

        # ── 5. Adapter loss: KL( teacher || adapter ).
        L_adapter = F.kl_div(adapter_log_probs, teacher_probs, reduction='batchmean')

        # ── 6. Retriever loss: soft mixture over cells weighted by adapter.
        #     Early training: adapter ~uniform → every cell gets gradient
        #       (matryoshka-style — all K prefixes become useful).
        #     Late training: adapter peaky → mostly the chosen cell gets
        #       gradient → specialization at that K.
        L_retriever = (adapter_probs * flat_losses).sum(dim=-1).mean()

        total_loss = L_retriever + self.adapter_weight * L_adapter

        # Diagnostics: marginal K_q and K_p distributions averaged over batch.
        with torch.no_grad():
            cell_grid = adapter_probs.view(B_q, n_K, n_K)  # [B_q, n_K_q, n_K_p]
            kq_dist = cell_grid.sum(dim=-1).mean(dim=0)    # [n_K]
            kp_dist = cell_grid.sum(dim=-2).mean(dim=0)    # [n_K]
            entropy = -(adapter_probs * adapter_log_probs).sum(dim=-1).mean()
            # Standalone sparse-only CE for logging (mirrors the legacy
            # `loss_sparse` field).  This is the InfoNCE on raw sparse scores
            # — it's NOT used for the gradient (sparse already contributes
            # via the per-cell dense+sparse fused score).  Reported just so
            # users can monitor sparse-only retrieval quality during training.
            if sparse_scores is not None:
                sparse_only_loss = F.cross_entropy(sparse_scores, labels)
            else:
                sparse_only_loss = torch.tensor(0.0, device=device)

        result = {
            'loss': total_loss,
            'loss_dense': L_retriever.detach(),
            'loss_sparse': sparse_only_loss.detach(),  # diagnostic only — not used for gradient
            'loss_adapter': L_adapter.detach(),
            'adapter_entropy': entropy.detach(),
        }
        # log marginals: kq_p1, kq_p2, kq_p4, kq_p8, kq_p16, kp_p1, ...
        for i, K in enumerate(K_opts):
            if i < kq_dist.shape[0]:
                result[f'kq_p{K}'] = kq_dist[i].detach()
                result[f'kp_p{K}'] = kp_dist[i].detach()
        return result

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
        passages layout: [pos_0, neg_0_0, …, neg_0_M, pos_1, neg_1_0, …]
        positive for query i is at index i * (1 + n_neg).

        num_denoise_steps == 1: single-pass fast path.
        num_denoise_steps >  1: multi-step denoising training.
          Runs the full denoising loop (each step decodes n_per_step tokens,
          storing frozen hidden states in repr_buf), then computes loss once
          at the final step. Gradient flows from the final mixed representation
          only — no dilution across steps. Uniform unmasking: n_per_step = K // n_steps.
        """
        B_q = query_input_ids.size(0)
        B_p = passage_input_ids.size(0)
        assert B_p % B_q == 0, f"B_p ({B_p}) must be divisible by B_q ({B_q})"
        n_paq = B_p // B_q
        device = query_input_ids.device
        labels = torch.arange(B_q, device=device) * n_paq
        n_steps = self.num_denoise_steps

        # ── Corruption augmentation (denoising-conditioned training) ─────────
        # Randomly mask text tokens (NOT gen MASKs) in passages to force robust,
        # diverse multi-vector representations.  Unique to diffusion: the model
        # was pretrained on this exact corruption — zero distribution shift.
        if self.training and self.corruption_rate > 0:
            import random as _rng
            t = _rng.uniform(0.0, self.corruption_rate)
            if t > 0.01:   # skip very small rates
                passage_input_ids = self._corrupt_text(
                    passage_input_ids, passage_attention_mask, t)

        # ── Soft-token multi-step (differentiable denoising) ─────────────────
        if self.soft_denoising and n_steps > 1:
            return self._soft_multistep_forward(
                query_input_ids, query_attention_mask,
                passage_input_ids, passage_attention_mask,
                query_content_ids, passage_content_ids,
                n_steps,
            )

        # ── K pre-encoder (two-stage encoding) ───────────────────────────────
        # Predict K per item BEFORE main encoder, slice each input to its K,
        # then run main encoder at variable-length batch.  True encoding
        # savings (vs the K-adapter which always encodes at K_max).
        if self.use_k_pre_encoder:
            return self._forward_with_k_pre_encoder(
                query_input_ids, query_attention_mask,
                passage_input_ids, passage_attention_mask,
                query_content_ids, passage_content_ids,
                labels,
            )

        # ── Single-step fast path ─────────────────────────────────────────────
        if n_steps <= 1:
            K_q, K_p = self.n_gen_q_tokens, self.n_gen_p_tokens
            n_tail = self._n_tail
            need_sparse = self.sparse_weight > 0

            # Optionally corrupt passages for denoising auxiliary loss
            if self.denoising_weight > 0:
                p_corrupted, p_denoise_targets, mask_ratio = self._apply_text_masking(
                    passage_input_ids, passage_attention_mask)
                p_ids_for_fwd = p_corrupted
            else:
                p_ids_for_fwd = passage_input_ids

            need_logits = need_sparse or (self.denoising_weight > 0)
            L_q, L_p = query_input_ids.size(1), p_ids_for_fwd.size(1)

            if K_q == K_p:
                # ── Symmetric path: concat Q+P into one _fwd (memory-efficient).
                if L_q < L_p:
                    pad = L_p - L_q
                    q_ids_cat = F.pad(query_input_ids, (pad, 0))       # left-pad
                    q_mask_cat = F.pad(query_attention_mask, (pad, 0))
                    p_ids_cat, p_mask_cat = p_ids_for_fwd, passage_attention_mask
                elif L_p < L_q:
                    pad = L_q - L_p
                    q_ids_cat, q_mask_cat = query_input_ids, query_attention_mask
                    p_ids_cat = F.pad(p_ids_for_fwd, (pad, 0))         # left-pad
                    p_mask_cat = F.pad(passage_attention_mask, (pad, 0))
                else:
                    q_ids_cat, q_mask_cat = query_input_ids, query_attention_mask
                    p_ids_cat, p_mask_cat = p_ids_for_fwd, passage_attention_mask

                all_ids = torch.cat([q_ids_cat, p_ids_cat], dim=0)
                all_mask = torch.cat([q_mask_cat, p_mask_cat], dim=0)
                all_hidden, all_logits = self._fwd(all_ids, all_mask,
                                                   need_logits=need_logits)
                L_all = all_ids.size(1)
                g_q_in_all = L_all - K_q - n_tail
                g_p_in_all = L_all - K_p - n_tail
                q_hidden, p_hidden = all_hidden[:B_q], all_hidden[B_q:]
                q_logits = all_logits[:B_q] if all_logits is not None else None
                p_logits = all_logits[B_q:] if all_logits is not None else None
            else:
                # ── Asymmetric path: K_q != K_p means different gen-block sizes,
                # so a concatenated forward would mis-align the slices.  Run
                # two separate _fwd calls (~2× compute but correct).
                q_hidden, q_logits = self._fwd(
                    query_input_ids, query_attention_mask,
                    need_logits=need_logits)
                p_hidden, p_logits = self._fwd(
                    p_ids_for_fwd, passage_attention_mask,
                    need_logits=need_logits)
                g_q_in_all = L_q - K_q - n_tail
                g_p_in_all = L_p - K_p - n_tail
                # No padding/concat: each side keeps its own length.
                L_all = max(L_q, L_p)                  # only used for denoising slice

            # ── Extract Q repr ─────────────────────────────────────────────
            q_repr_hidden = q_hidden[:, g_q_in_all:g_q_in_all + K_q, :].float()
            q_quotation_emb = q_hidden[:, g_q_in_all - 1, :].float()
            q_sparse_max = None
            if need_sparse and q_logits is not None:
                # Monotonic trick: max in bf16 first → log1p(relu()) on [B, V].
                q_sparse_max = torch.log1p(torch.relu(
                    q_logits[:, g_q_in_all:g_q_in_all + K_q, :].max(dim=1).values))
            if self.normalize:
                q_quotation_emb = F.normalize(q_quotation_emb, p=2, dim=-1)
                q_repr_hidden = F.normalize(q_repr_hidden, p=2, dim=-1)
            q_repr = {'repr_hidden': q_repr_hidden, 'quotation_emb': q_quotation_emb}
            if q_sparse_max is not None:
                if query_content_ids is not None:
                    q_sparse_max = filter_sparse(q_sparse_max, query_content_ids)
                q_repr['sparse_acts'] = q_sparse_max

            # ── Extract P repr ─────────────────────────────────────────────
            p_repr_hidden = p_hidden[:, g_p_in_all:g_p_in_all + K_p, :].float()
            p_quotation_emb = p_hidden[:, g_p_in_all - 1, :].float()
            p_sparse_max = None
            if need_sparse and p_logits is not None:
                # Monotonic trick: max in bf16 first → log1p(relu()) on [B, V].
                p_sparse_max = torch.log1p(torch.relu(
                    p_logits[:, g_p_in_all:g_p_in_all + K_p, :].max(dim=1).values))
            if self.normalize:
                p_quotation_emb = F.normalize(p_quotation_emb, p=2, dim=-1)
                p_repr_hidden = F.normalize(p_repr_hidden, p=2, dim=-1)
            p_repr = {'repr_hidden': p_repr_hidden, 'quotation_emb': p_quotation_emb}
            if p_sparse_max is not None:
                if passage_content_ids is not None:
                    p_sparse_max = filter_sparse(p_sparse_max, passage_content_ids)
                p_repr['sparse_acts'] = p_sparse_max

            # ── Cross-GPU negative sharing ────────────────────────────────
            q_repr = self._gather_repr(q_repr)
            p_repr = self._gather_repr(p_repr)
            # Recompute labels for gathered batch
            B_q_all = q_repr['repr_hidden'].size(0)
            B_p_all = p_repr['repr_hidden'].size(0)
            n_paq_g = B_p_all // B_q_all
            labels_g = torch.arange(B_q_all, device=device) * n_paq_g

            # ── Loss ──────────────────────────────────────────────────────
            loss_dict = self.compute_loss(q_repr, p_repr, labels_g)
            if self.debug_dense_metrics:
                loss_dict.update(self._dense_debug_stats(
                    q_repr, p_repr, labels_g,
                    q_raw=q_repr_hidden, p_raw=p_repr_hidden,
                ))

            # Scale loss to counter DDP gradient averaging
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                loss_dict['loss'] = loss_dict['loss'] * torch.distributed.get_world_size()

            if self.denoising_weight > 0:
                # Denoising targets were created from original passage length;
                # if passages were left-padded to match queries, slice
                # accordingly.  In asymmetric mode (separate forward) no
                # padding was applied, so the slice is a no-op (dn_start=0).
                p_seq_len = p_logits.size(1) if p_logits is not None else L_p
                dn_start = p_seq_len - L_p
                p_logits_dn = p_logits[:, dn_start:, :]
                denoise_loss = self.compute_denoising_loss(
                    p_logits_dn, p_denoise_targets, mask_ratio)
                loss_dict['loss'] = loss_dict['loss'] + self.denoising_weight * denoise_loss
                loss_dict['loss_denoising'] = denoise_loss.detach()

            return loss_dict

        # ── Multi-step: denoising training ───────────────────────────────────
        K_q, K_p = self.n_gen_q_tokens, self.n_gen_p_tokens
        n_tail = self._n_tail
        n_per_step_q = max(1, K_q // n_steps)
        n_per_step_p = max(1, K_p // n_steps)
        device = query_input_ids.device
        H = self.hidden_size
        V = self.backbone.config.vocab_size
        L_q = query_input_ids.size(1)
        L_p = passage_input_ids.size(1)

        # Pre-compute 4D masks once (only for models that need them)
        q_mask_4d = self._build_4d_mask(query_input_ids.size(1), query_attention_mask) if (self.adapter.needs_4d_mask() if self.adapter else True) else None
        p_mask_4d = self._build_4d_mask(passage_input_ids.size(1), passage_attention_mask) if (self.adapter.needs_4d_mask() if self.adapter else True) else None

        q_curr = query_input_ids.clone()
        p_curr = passage_input_ids.clone()
        need_sparse = self.sparse_weight > 0

        # Frozen buffers for positions already decoded in prior steps
        q_repr_buf = torch.zeros(B_q, K_q, H, device=device)
        p_repr_buf = torch.zeros(B_p, K_p, H, device=device)
        q_decoded_mask = torch.zeros(B_q, K_q, dtype=torch.bool, device=device)
        p_decoded_mask = torch.zeros(B_p, K_p, dtype=torch.bool, device=device)
        q_sparse_decoded = torch.zeros(B_q, V, device=device) if need_sparse else None
        p_sparse_decoded = torch.zeros(B_p, V, device=device) if need_sparse else None

        progressive = self.progressive_step_weight > 0

        final_loss: Dict[str, torch.Tensor] = {}
        progressive_loss_sum = torch.tensor(0.0, device=device)

        def _build_repr_at_step(h, logits, repr_buf, decoded_mask, sparse_decoded,
                                K_side: int,
                                content_ids=None, use_fresh=False):
            """Build retrieval representation from current hidden states.
            Used for both intermediate (progressive) and final steps.
            Snapshot buffers with .clone() to isolate from later in-place updates.

            K_side: per-side K (n_gen_q_tokens for queries, n_gen_p_tokens
            for passages — needed under asymmetric configs).

            use_fresh: if True, ALL K positions use current hidden states
            (ignoring frozen repr_buf). Lets all tokens benefit from
            fully-decoded context at the final step.
            """
            Ls = h.size(1)
            gs = Ls - K_side - n_tail  # left-padded: gen block then structural tail
            curr_gen_h = h[:, gs:gs + K_side, :].float()  # [B_loc, K_side, H]

            decoded_snap = decoded_mask.clone()
            repr_snap = repr_buf.detach().clone()

            mask_sparse = None
            if need_sparse and logits is not None:
                sp_all = torch.log(1.0 + torch.relu(logits[:, gs:gs + K_side, :].float()))
                if use_fresh:
                    mask_sparse = sp_all.max(dim=1).values
                else:
                    is_mask = (~decoded_snap).float().unsqueeze(-1)
                    mask_sparse = (sp_all * is_mask).max(dim=1).values

            if use_fresh:
                mixed = curr_gen_h
            else:
                mixed = torch.where(decoded_snap.unsqueeze(-1), repr_snap, curr_gen_h)
            quotation_emb = h[:, gs - 1, :].float()

            if self.normalize:
                mixed = F.normalize(mixed, p=2, dim=-1)
                quotation_emb = F.normalize(quotation_emb, p=2, dim=-1)

            result = {'repr_hidden': mixed, 'quotation_emb': quotation_emb}
            if need_sparse and sparse_decoded is not None:
                sparse_snap = sparse_decoded.detach().clone()
                combined_sparse = torch.max(sparse_snap, mask_sparse)
                if content_ids is not None:
                    combined_sparse = filter_sparse(combined_sparse, content_ids)
                result['sparse_acts'] = combined_sparse
            return result

        q_g = L_q - K_q - n_tail  # left-padded: gen block starts here
        p_g = L_p - K_p - n_tail

        for step in range(n_steps):
            is_last = (step == n_steps - 1)

            q_h, q_logits = self._fwd(q_curr, query_attention_mask,
                                      need_logits=True, mask_4d=q_mask_4d)
            p_h, p_logits = self._fwd(p_curr, passage_attention_mask,
                                      need_logits=True, mask_4d=p_mask_4d)

            if progressive or is_last:
                _fresh = self.use_fresh_final and is_last
                q_repr_s = _build_repr_at_step(
                    q_h, q_logits,
                    q_repr_buf, q_decoded_mask, q_sparse_decoded, K_q,
                    content_ids=query_content_ids, use_fresh=_fresh)
                p_repr_s = _build_repr_at_step(
                    p_h, p_logits,
                    p_repr_buf, p_decoded_mask, p_sparse_decoded, K_p,
                    content_ids=passage_content_ids, use_fresh=_fresh)

                # Cross-GPU gather for multi-step
                q_repr_s = self._gather_repr(q_repr_s)
                p_repr_s = self._gather_repr(p_repr_s)
                B_q_g = q_repr_s['repr_hidden'].size(0)
                B_p_g = p_repr_s['repr_hidden'].size(0)
                labels_ms = torch.arange(B_q_g, device=device) * (B_p_g // B_q_g)
                step_loss_dict = self.compute_loss(q_repr_s, p_repr_s, labels_ms)

                if is_last:
                    final_loss = step_loss_dict
                    if self.debug_dense_metrics:
                        final_loss.update(self._dense_debug_stats(q_repr_s, p_repr_s, labels_ms))
                elif progressive:
                    step_weight = (step + 1) / n_steps
                    progressive_loss_sum = progressive_loss_sum + step_weight * step_loss_dict['loss']

            if not is_last:
                q_curr, q_newly = self._unmask_step(
                    q_curr, q_logits, K_q, n_per_step_q)
                p_curr, p_newly = self._unmask_step(
                    p_curr, p_logits, K_p, n_per_step_p)

                with torch.no_grad():
                    for i, pos_list in enumerate(q_newly):
                        for tok_pos in pos_list:
                            q_repr_buf[i, tok_pos] = q_h[i, q_g + tok_pos].detach()
                            q_decoded_mask[i, tok_pos] = True
                            if q_sparse_decoded is not None and q_logits is not None:
                                q_sparse_decoded[i] = torch.max(q_sparse_decoded[i],
                                    torch.log(1.0 + torch.relu(q_logits[i, q_g + tok_pos])).detach())
                    for i, pos_list in enumerate(p_newly):
                        for tok_pos in pos_list:
                            p_repr_buf[i, tok_pos] = p_h[i, p_g + tok_pos].detach()
                            p_decoded_mask[i, tok_pos] = True
                            if p_sparse_decoded is not None and p_logits is not None:
                                p_sparse_decoded[i] = torch.max(p_sparse_decoded[i],
                                    torch.log(1.0 + torch.relu(p_logits[i, p_g + tok_pos])).detach())

        # Add progressive step loss to final loss
        if progressive and final_loss:
            final_loss['loss'] = final_loss['loss'] + self.progressive_step_weight * progressive_loss_sum
            final_loss['loss_progressive'] = progressive_loss_sum.detach()

        # Scale loss for DDP (multi-step path)
        if final_loss and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            final_loss['loss'] = final_loss['loss'] * torch.distributed.get_world_size()

        # Denoising auxiliary for multi-step (same as single-step: separate corrupted forward)
        if self.denoising_weight > 0 and final_loss:
            p_corrupted, p_denoise_targets, mask_ratio = self._apply_text_masking(
                passage_input_ids, passage_attention_mask)
            _, p_logits_dn = self._fwd(p_corrupted, passage_attention_mask,
                                        need_logits=True, mask_4d=p_mask_4d)
            denoise_loss = self.compute_denoising_loss(p_logits_dn, p_denoise_targets, mask_ratio)
            final_loss['loss'] = final_loss['loss'] + self.denoising_weight * denoise_loss
            final_loss['loss_denoising'] = denoise_loss.detach()

        return final_loss

    # ----------------------------------------------------------------
    # Save / load helpers
    # ----------------------------------------------------------------

    def _save_retriever_config(self, output_dir: str):
        import json, os
        config = {
            'model_type': self.model_type,
            'mask_token_id': self.mask_token_id,
            'hidden_size': self.hidden_size,
            'max_length': self.max_length,
            'n_gen_tokens': self.n_gen_tokens,
            'n_gen_q_tokens': self.n_gen_q_tokens,
            'n_gen_p_tokens': self.n_gen_p_tokens,
            'temperature': self.temperature,
            'flops_weight': 0.0,  # deprecated, kept for backwards compat
            'num_denoise_steps': self.num_denoise_steps,
            'sparse_weight': self.sparse_weight,
            'normalize': self.normalize,
            'query_prefix_ids': self._query_prefix_ids,
            'query_suffix_ids': self._query_suffix_ids,
            'passage_prefix_ids': self._passage_prefix_ids,
            'passage_suffix_ids': self._passage_suffix_ids,
            'lora_rank': getattr(self, 'lora_rank', 0),
            'lora_alpha': getattr(self, 'lora_alpha', 64),
            'use_eos': self.use_eos,
            'n_tail': self._n_tail,
            # K-adapter persistence
            'use_k_adapter': self.use_k_adapter,
            'adapter_weight': self.adapter_weight,
            'teacher_temperature': self.teacher_temperature,
            'k_adapter_options': list(self.k_adapter_options),
            # K pre-encoder persistence
            'use_k_pre_encoder': self.use_k_pre_encoder,
            'gumbel_temperature': self.gumbel_temperature,
            'k_cost_lambda': self.k_cost_lambda,
            'k_pre_encoder_options': list(self.k_pre_encoder_options),
        }
        with open(os.path.join(output_dir, 'retriever_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        # Save adapter state separately (PEFT save_pretrained only saves LoRA;
        # the small KAdapter MLP isn't a LoRA module so we persist it ourselves).
        if self.use_k_adapter and self.k_adapter is not None:
            torch.save(self.k_adapter.state_dict(),
                       os.path.join(output_dir, 'k_adapter.bin'))
        if self.use_k_pre_encoder and self.k_pre_encoder_q is not None:
            torch.save({'q': self.k_pre_encoder_q.state_dict(),
                        'p': self.k_pre_encoder_p.state_dict()},
                       os.path.join(output_dir, 'k_pre_encoder.bin'))

    @classmethod
    def load(cls, model_dir: str, **fallback_kwargs) -> 'TrainableDiffusionRetriever':
        """Load a fine-tuned TrainableDiffusionRetriever from a saved directory.

        If retriever_config.json is missing (e.g. mid-training checkpoint or old checkpoint
        with matryoshka_config.json), falls back to from_backbone() using fallback_kwargs:
        model_type, query_prompt, passage_prompt, n_gen_tokens,
        sparse_weight, max_length.
        """
        import json
        from transformers import AutoTokenizer

        # Support both new name and old name (matryoshka_config.json) for backwards compat
        config_path = Path(model_dir) / 'retriever_config.json'
        if not config_path.exists():
            config_path = Path(model_dir) / 'matryoshka_config.json'
        if not config_path.exists():
            model_type = fallback_kwargs.get('model_type')
            query_prompt = fallback_kwargs.get('query_prompt')
            passage_prompt = fallback_kwargs.get('passage_prompt')
            if not model_type or not query_prompt or not passage_prompt:
                raise FileNotFoundError(
                    f"No retriever_config.json in {model_dir}. "
                    "Pass model_type, query_prompt, passage_prompt as fallback kwargs.")

            _fallback_adapter = get_adapter(model_type)
            source_model = (fallback_kwargs.get('original_model')
                            or _fallback_adapter.hub_model_name)
            logger.info(f"No retriever_config.json — loading architecture from {source_model}, "
                        f"weights from {model_dir}")

            # Load state dict first so we can detect LoRA rank before building model
            checkpoint_dir = Path(model_dir)
            weight_file = checkpoint_dir / 'model.safetensors'
            bin_files = sorted(checkpoint_dir.glob('model-*.safetensors'))
            bin_pt = sorted(checkpoint_dir.glob('pytorch_model*.bin'))

            state_dict = None
            if weight_file.exists() or bin_files:
                from safetensors.torch import load_file
                if weight_file.exists():
                    state_dict = load_file(str(weight_file))
                else:
                    state_dict = {}
                    for f in bin_files:
                        state_dict.update(load_file(str(f)))
            elif bin_pt:
                state_dict = torch.load(str(bin_pt[0]), map_location='cpu')

            # Detect LoRA rank from key names (keys look like backbone.base_model.model.*)
            detected_lora_rank = fallback_kwargs.get('lora_rank', 0)
            if state_dict is not None and detected_lora_rank == 0:
                for k, v in state_dict.items():
                    if 'lora_A.default.weight' in k:
                        detected_lora_rank = v.shape[0]
                        logger.info(f"Detected LoRA rank {detected_lora_rank} from checkpoint keys")
                        break

            model = cls.from_backbone(
                model_name=source_model,
                model_type=model_type,
                query_prompt=query_prompt,
                passage_prompt=passage_prompt,
                n_gen_tokens=fallback_kwargs.get('n_gen_tokens', 4),
                n_gen_q_tokens=fallback_kwargs.get('n_gen_q_tokens', None),
                n_gen_p_tokens=fallback_kwargs.get('n_gen_p_tokens', None),
                temperature=fallback_kwargs.get('temperature', 0.02),
                num_denoise_steps=fallback_kwargs.get('num_denoise_steps', None),
                sparse_weight=fallback_kwargs.get('sparse_weight', 1.0),
                normalize=fallback_kwargs.get('normalize', True),
                max_length=fallback_kwargs.get('max_length', 256),
                lora_rank=detected_lora_rank,
                device_map='auto',
            )

            if state_dict is not None:
                # Load full state dict directly into the model (keys have backbone. prefix)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                backbone_keys_loaded = sum(1 for k in state_dict if k.startswith('backbone.'))
                logger.info(f"Loaded checkpoint weights from {model_dir} "
                            f"(lora_rank={detected_lora_rank}, "
                            f"{backbone_keys_loaded} backbone keys, "
                            f"{len(missing)} missing, {len(unexpected)} unexpected)")
            else:
                logger.warning(f"No weight file found in {model_dir} — using original model weights")

            # Merge LoRA weights into base model for faster inference
            if detected_lora_rank > 0 and hasattr(model.backbone, 'merge_and_unload'):
                model.backbone = model.backbone.merge_and_unload()
                logger.info("Merged LoRA adapters for inference")

            return model

        with open(config_path) as f:
            cfg = json.load(f)

        model_type = cfg.get('model_type', 'dream')
        adapter = get_adapter(model_type)

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer.padding_side = 'left'

        load_weights_separately = False

        try:
            backbone = adapter.load_backbone(model_dir, device_map='auto')
        except ValueError:
            # model_dir has no config.json — load architecture from Hub
            logger.info(f"No valid config.json in {model_dir} — "
                        f"loading architecture from {adapter.hub_model_name}")
            backbone = adapter.load_backbone(adapter.hub_model_name, device_map='auto')
            load_weights_separately = True

        # Detect if model.safetensors was saved as a full PEFT state dict by DeepSpeed
        _ckpt = Path(model_dir) / 'model.safetensors'
        if _ckpt.exists():
            from safetensors import safe_open as _so
            with _so(str(_ckpt), framework='pt') as _f:
                _sample = list(_f.keys())[:8]
            if any('base_model.model' in k for k in _sample):
                logger.info("Detected PEFT/LoRA state dict — "
                            "applying LoRA to backbone before loading weights")
                _lora_rank = cfg.get('lora_rank', 0)
                _lora_alpha = cfg.get('lora_alpha', 64)
                if _lora_rank == 0:
                    from safetensors.torch import load_file as _lf
                    _peek = _lf(str(_ckpt))
                    _la_keys = [k for k in _peek if 'lora_A.default.weight' in k]
                    if _la_keys:
                        _lora_rank = _peek[_la_keys[0]].shape[0]
                if _lora_rank > 0 and not getattr(backbone, 'peft_config', None):
                    from peft import get_peft_model
                    lora_cfg = adapter.get_lora_config(_lora_rank, _lora_alpha)
                    backbone = get_peft_model(backbone, lora_cfg)
                    logger.info(f"Applied LoRA rank={_lora_rank} alpha={_lora_alpha}")
                elif getattr(backbone, 'peft_config', None):
                    logger.info("Backbone already has PEFT — skipping duplicate LoRA")
                load_weights_separately = True

        model = cls(
            backbone=backbone,
            tokenizer=tokenizer,
            mask_token_id=cfg['mask_token_id'],
            hidden_size=cfg['hidden_size'],
            query_prefix_ids=cfg['query_prefix_ids'],
            query_suffix_ids=cfg['query_suffix_ids'],
            passage_prefix_ids=cfg['passage_prefix_ids'],
            passage_suffix_ids=cfg['passage_suffix_ids'],
            max_length=cfg['max_length'],
            n_gen_tokens=cfg['n_gen_tokens'],
            n_gen_q_tokens=cfg.get('n_gen_q_tokens'),  # back-compat: missing → fall through to n_gen_tokens
            n_gen_p_tokens=cfg.get('n_gen_p_tokens'),
            temperature=cfg['temperature'],
            num_denoise_steps=cfg['num_denoise_steps'],
            sparse_weight=cfg['sparse_weight'],
            normalize=cfg['normalize'],
            flash_attn=adapter.flash_attn,
            use_eos=cfg.get('use_eos', False),
            # K-adapter restore (back-compat: keys missing → defaults disable)
            use_k_adapter=cfg.get('use_k_adapter', False),
            adapter_weight=cfg.get('adapter_weight', 1.0),
            teacher_temperature=cfg.get('teacher_temperature', 1.0),
            k_adapter_options=tuple(cfg['k_adapter_options'])
                              if cfg.get('k_adapter_options') is not None else None,
            # K pre-encoder restore
            use_k_pre_encoder=cfg.get('use_k_pre_encoder', False),
            gumbel_temperature=cfg.get('gumbel_temperature', 1.0),
            k_cost_lambda=cfg.get('k_cost_lambda', 0.001),
            k_pre_encoder_options=tuple(cfg['k_pre_encoder_options'])
                                  if cfg.get('k_pre_encoder_options') is not None else None,
        )
        # Load adapter weights if a saved file exists.
        if model.use_k_adapter and model.k_adapter is not None:
            adapter_path = Path(model_dir) / 'k_adapter.bin'
            if adapter_path.exists():
                model.k_adapter.load_state_dict(
                    torch.load(str(adapter_path), map_location='cpu'))
                logger.info(f"Loaded KAdapter state from {adapter_path}")
            else:
                logger.warning(f"use_k_adapter=True but {adapter_path} missing; "
                               f"adapter starts from random init.")
        # Load pre-encoder weights if a saved file exists.
        if model.use_k_pre_encoder and model.k_pre_encoder_q is not None:
            pe_path = Path(model_dir) / 'k_pre_encoder.bin'
            if pe_path.exists():
                pe_state = torch.load(str(pe_path), map_location='cpu')
                model.k_pre_encoder_q.load_state_dict(pe_state['q'])
                model.k_pre_encoder_p.load_state_dict(pe_state['p'])
                logger.info(f"Loaded KPreEncoder state from {pe_path}")
            else:
                logger.warning(f"use_k_pre_encoder=True but {pe_path} missing; "
                               f"pre-encoder starts from random init.")
        # Restore _n_tail from config.  Old checkpoints (before structural tail)
        # don't have 'n_tail' — infer from use_eos for backward compat.
        if 'n_tail' in cfg:
            model._n_tail = cfg['n_tail']
            model._tail_ids = model._build_tail_ids()
        else:
            # Legacy: use_eos=True → 1 tail token (EOS only), use_eos=False → 0
            model._n_tail = 1 if cfg.get('use_eos', False) else 0
            eos_id = model.tokenizer.eos_token_id
            model._tail_ids = [eos_id] if model._n_tail == 1 else []
            logger.info(f"Legacy checkpoint: n_tail={model._n_tail} (from use_eos={cfg.get('use_eos', False)})")
        model.model_type = model_type
        model.adapter = adapter

        # Hook for efficient hidden state extraction
        model._last_hidden: Dict[str, torch.Tensor] = {}
        model._hook_registered = adapter.register_hidden_hook(
            backbone, model._last_hidden)

        if load_weights_separately:
            checkpoint_dir = Path(model_dir)
            weight_file = checkpoint_dir / 'model.safetensors'
            bin_files = sorted(checkpoint_dir.glob('model-*.safetensors'))
            bin_pt = sorted(checkpoint_dir.glob('pytorch_model*.bin'))
            if weight_file.exists() or bin_files:
                from safetensors.torch import load_file
                state_dict = load_file(str(weight_file)) if weight_file.exists() else {}
                for f in bin_files:
                    state_dict.update(load_file(str(f)))
                backbone_dict = {k[len('backbone.'):]: v for k, v in state_dict.items()
                                 if k.startswith('backbone.')}
                model.backbone.load_state_dict(backbone_dict or state_dict, strict=False)
                logger.info(f"Loaded fine-tuned weights from {model_dir}")
            elif bin_pt:
                sd = torch.load(str(bin_pt[0]), map_location='cpu')
                backbone_dict = {k[len('backbone.'):]: v for k, v in sd.items()
                                 if k.startswith('backbone.')}
                model.backbone.load_state_dict(backbone_dict or sd, strict=False)
                logger.info(f"Loaded fine-tuned weights from {model_dir}")
            else:
                logger.warning(f"No weight file found in {model_dir} — using Hub weights")

        # Merge LoRA adapters into the base weights for inference. Training
        # keeps them separate for gradient flow; at inference each LoRA adds
        # per-layer overhead otherwise.
        #
        # Two shapes seen in saved checkpoints:
        #   (a) `model.backbone` is a PeftModel wrapper → call
        #       `merge_and_unload()` directly.
        #   (b) `model.backbone` is the underlying DreamModel / LLaDAModel
        #       with `LoraLinear` modules still in place (the top-level
        #       PeftModel wrapper was dropped during save).  `merge_and_unload`
        #       is missing but each inner `LoraLinear` still exposes `.merge()`
        #       which folds its `lora_A @ lora_B` into `base_layer.weight` in
        #       place and flips `merged=True`, disabling the LoRA forward path.
        #       We walk the module tree and merge case-by-case.
        if hasattr(model.backbone, 'merge_and_unload'):
            try:
                model.backbone = model.backbone.merge_and_unload()
                logger.info("Merged LoRA adapters into base backbone for inference")
            except Exception as exc:
                logger.warning(f"merge_and_unload failed ({exc}); "
                               f"continuing with un-merged LoRA (slower inference)")
        else:
            merged_modules = 0
            for _mod in model.backbone.modules():
                if (hasattr(_mod, 'lora_A') and hasattr(_mod, 'merge')
                        and callable(getattr(_mod, 'merge', None))
                        and not getattr(_mod, 'merged', False)):
                    try:
                        _mod.merge()
                        merged_modules += 1
                    except Exception as exc:
                        logger.warning(f"per-module LoRA merge failed ({exc}) "
                                       f"— continuing with un-merged layer")
            if merged_modules > 0:
                logger.info(f"Merged {merged_modules} LoraLinear modules into "
                            f"base weights for inference")

        logger.info(f"Loaded TrainableDiffusionRetriever from {model_dir}")
        return model

    def save(self, output_dir: str):
        import os
        os.makedirs(output_dir, exist_ok=True)

        backbone = self.backbone
        if hasattr(backbone, 'save_pretrained'):
            backbone.save_pretrained(output_dir)
        else:
            backbone.base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self._save_retriever_config(output_dir)
        logger.info(f"Saved to {output_dir}")

    @property
    def config(self):
        # HF Trainer / DeepSpeed expect model.config to exist.
        return self.backbone.config

    def gradient_checkpointing_enable(self, **kwargs):
        if self.adapter:
            self.adapter.enable_gradient_checkpointing(self.backbone, **kwargs)
        else:
            self.backbone.gradient_checkpointing_enable(**kwargs)
