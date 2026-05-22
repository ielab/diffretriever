"""
DiffRetriever — Dream backbone (zero-shot).

Uses Dream-v0-Instruct-7B (a discrete diffusion LLM initialised from Qwen2.5)
as the backbone for masked-position prediction retrieval (see paper §3.2).
The model appends K masked positions after the retrieval prompt and reads
K hidden states + K logit vectors from a single bidirectional forward pass.
Like LLaDA, Dream uses bidirectional attention and a mask token for
masked-position denoising; unlike LLaDA, Dream uses full-sequence diffusion
without block scheduling.

Two encoding strategies:
1. Clean encoding: Single forward pass, all tokens visible (bidirectional).
2. PromptReps encoding: Wraps text with a prompt template, appends [MASK] token(s),
   denoises them, uses hidden states at [MASK] positions as embeddings.

Reference: Dream-org/Dream-v0-Instruct-7B
- Qwen2-based architecture, 8B params
- Mask token: <|mask|> = 151666
- Hidden size: 3584, Vocab: 152064, 28 layers
- Load with AutoModel.from_pretrained (NOT AutoModelForCausalLM)
- Forward returns MaskedLMOutput with .logits and .hidden_states
- is_causal=False — bidirectional attention
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List, Dict, Literal
from pathlib import Path
import logging

_NUM_WORDS = ['one', 'two', 'three', 'four', 'five',
              'six', 'seven', 'eight', 'nine', 'ten']

from .sparse_utils import get_content_token_ids, filter_sparse

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151666  # Dream's [MASK] token id (<|mask|>)


class DreamRetriever(nn.Module):
    """
    Dense retriever built on Dream's discrete diffusion process.

    Supports two encoding modes:
    - clean: standard bidirectional encoding
    - promptreps: Dream PromptReps — prompt + [MASK] → denoise → embedding

    Embeddings are the raw hidden states from the backbone (no projection).
    """

    def __init__(
        self,
        model_name: str = "Dream-org/Dream-v0-Instruct-7B",
        max_length: int = 256,
        pooling: Literal["mean", "weighted_mean", "last", "attention"] = "mean",
        normalize: bool = True,
        freeze_backbone: bool = False,
        mask_token_id: int = MASK_TOKEN_ID,
        num_repr_tokens: int = 1,
        num_denoise_steps: int = 1,
        query_prompt: str = "",
        passage_prompt: str = "",
        use_quotation_token: bool = True,
        n_gen_tokens: int = 0,
        filter_structural: bool = False,
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self.mask_token_id = mask_token_id
        self.num_repr_tokens = num_repr_tokens
        self.num_denoise_steps = num_denoise_steps
        self.use_quotation_token = use_quotation_token
        self.n_gen_tokens = n_gen_tokens
        self._n_tail = 3  # extra MASK tokens appended to absorb structural closing tokens
        self.filter_structural = filter_structural

        # Load Dream model (uses AutoModel, not AutoModelForCausalLM)
        logger.info(f"Loading Dream from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self._structural_token_ids = self._compute_structural_ids() if filter_structural else frozenset()
        try:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                attn_implementation=attn_implementation,
            )
            logger.info(f"Using {attn_implementation}")
        except (ValueError, ImportError):
            self.backbone = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
            logger.info("Flash Attention 2 not available, using default attention")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Dream backbone frozen")

        # Build prefix/suffix token IDs for promptreps (cache once).
        q_yaml = self._load_yaml_prompt(query_prompt)
        p_yaml = self._load_yaml_prompt(passage_prompt)
        if q_yaml is None or p_yaml is None:
            raise ValueError(
                "--query_prompt and --passage_prompt must be paths to valid YAML files. "
                f"Got: query_prompt={query_prompt!r}, passage_prompt={passage_prompt!r}"
            )

        q_yaml = dict(q_yaml)
        q_yaml['user_suffix'] = self._adapt_prompt_for_k(q_yaml.get('user_suffix', ''), n_gen_tokens)
        q_yaml['assistant_prefix'] = self._adapt_prompt_for_k(q_yaml.get('assistant_prefix', ''), n_gen_tokens)
        self._query_prefix_ids, self._query_suffix_ids = self._build_chat_prompt_ids(q_yaml)

        p_yaml = dict(p_yaml)
        p_yaml['user_suffix'] = self._adapt_prompt_for_k(p_yaml.get('user_suffix', ''), n_gen_tokens)
        p_yaml['assistant_prefix'] = self._adapt_prompt_for_k(p_yaml.get('assistant_prefix', ''), n_gen_tokens)
        self._passage_prefix_ids, self._passage_suffix_ids = self._build_chat_prompt_ids(p_yaml)

        # Check if Flash Attention 2 is active
        self._flash_attn = getattr(
            self.backbone.config, '_attn_implementation', None
        ) == 'flash_attention_2'
        if self._flash_attn:
            logger.info("Flash Attention 2 active — will use 2D attention masks (bidirectional)")

        # Determine hidden size
        self.hidden_size = getattr(
            self.backbone.config, 'hidden_size',
            getattr(self.backbone.config, 'd_model', 3584)
        )

        # Attention pooling (optional)
        if pooling == "attention":
            self.attn_pool = nn.Sequential(nn.Linear(self.hidden_size, 1))

    @staticmethod
    def _load_yaml_prompt(path: str) -> Optional[dict]:
        """Load a YAML prompt file. Returns None if path is empty or not a .yaml/.yml file."""
        if not path:
            return None
        p = Path(path)
        if p.exists() and p.suffix in ('.yaml', '.yml'):
            import yaml
            return yaml.safe_load(p.read_text())
        return None

    def _build_chat_prompt_ids(self, yaml_dict: dict):
        """Build (prefix_ids, suffix_ids) from a YAML prompt dict using the chat template.

        Uses a sentinel to split the formatted template at the text insertion point:
          prefix_ids: system prompt + user turn start + user_prefix
          suffix_ids: user_suffix + user turn end + assistant turn start + assistant_prefix

        The sequence becomes:
          [prefix_ids] [text_ids] [suffix_ids] [MASK×K]
        where the last token of suffix_ids is the quotation '"' from assistant_prefix.
        """
        system = yaml_dict.get('system', '')
        user_prefix = yaml_dict.get('user_prefix', '')
        user_suffix = yaml_dict.get('user_suffix', '')
        assistant_prefix = yaml_dict.get('assistant_prefix', '')

        if yaml_dict.get('template') == 'none':
            prefix_ids = self.tokenizer.encode(user_prefix, add_special_tokens=False)
            suffix_ids = self.tokenizer.encode(user_suffix + assistant_prefix, add_special_tokens=False)
            logger.info(f"No chat template: prefix={len(prefix_ids)} tokens, suffix={len(suffix_ids)} tokens")
            logger.info(f"  user_suffix: {user_suffix!r}")
            return prefix_ids, suffix_ids

        SENTINEL = "XSENTINELX"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_prefix + SENTINEL + user_suffix})

        full_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if SENTINEL not in full_str:
            raise ValueError(f"Sentinel not found in chat template output: {full_str!r}")

        pre_str, post_str = full_str.split(SENTINEL, 1)
        prefix_ids = self.tokenizer.encode(pre_str, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(post_str + assistant_prefix, add_special_tokens=False)

        logger.info(f"Chat prompt: prefix={len(prefix_ids)} tokens, suffix={len(suffix_ids)} tokens "
                    f"(user_suffix + chat markers + assistant_prefix)")
        logger.info(f"  user_suffix: {user_suffix!r}")
        logger.info(f"  assistant_prefix: {assistant_prefix!r}")
        return prefix_ids, suffix_ids

    @staticmethod
    def _adapt_prompt_for_k(text: str, k: int) -> str:
        """Rewrite number words in a prompt string to match k.

        Prompt files are written for k=1 ("one word ... The word is").
        For k>1 this replaces the count with the appropriate number word
        and fixes singular/plural in both user_suffix and assistant_prefix.

        Examples:
          k=1: unchanged  → "... one word ... The word is: "
          k=4: adapted    → "... four words ... The words are: "
        """
        if k <= 1 or not text:
            return text
        count = _NUM_WORDS[k - 1] if k <= len(_NUM_WORDS) else str(k)
        # Replace any existing number word + "word(s)" → "{count} words"
        result = re.sub(
            r'\b(?:' + '|'.join(_NUM_WORDS) + r')\b(\s+words?)',
            lambda m: f'{count} words',
            text,
        )
        # Fix "word is" → "words are" (covers "The word is", "your word is", etc.)
        result = re.sub(r'\bword is\b', 'words are', result)
        return result

    # Keep old name as alias for backward compatibility
    _adapt_suffix_for_k = _adapt_prompt_for_k

    def _exact_token_id(self, token: str) -> Optional[int]:
        tok_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, 'unk_token_id', None)
        if tok_id is not None and tok_id >= 0 and tok_id != unk_id:
            return tok_id
        return None

    def _single_token_text_id(self, text: str) -> Optional[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _compute_structural_ids(self) -> frozenset:
        """Token IDs whose decoded text contains '"' — structural closing tokens."""
        structural = set()
        for tok_str in self.tokenizer.get_vocab():
            decoded = self.tokenizer.convert_tokens_to_string([tok_str])
            if '"' in decoded:
                structural.add(self.tokenizer.convert_tokens_to_ids(tok_str))
        return frozenset(structural)

    def _build_tail_ids(self, n_tail: int) -> List[int]:
        if n_tail <= 0:
            return []
        quote_id = self._single_token_text_id('"')
        im_end_id = self._exact_token_id('<|im_end|>')
        eos_id = self.tokenizer.eos_token_id
        tail_ids: List[int] = []
        if quote_id is not None:
            tail_ids.append(quote_id)
        if im_end_id is not None and im_end_id != eos_id:
            tail_ids.append(im_end_id)
        while len(tail_ids) < n_tail:
            tail_ids.append(eos_id)
        return tail_ids[:n_tail]

    def _tokenize_promptreps(
        self, texts: List[str], is_query: bool,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts with prefix + text + suffix + repr/tail block."""
        prefix_ids = self._query_prefix_ids if is_query else self._passage_prefix_ids
        suffix_ids = self._query_suffix_ids if is_query else self._passage_suffix_ids

        n_tail = self._n_tail if self.n_gen_tokens > 0 else 0
        max_text_len = self.max_length - len(prefix_ids) - len(suffix_ids)

        text_encodings = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_text_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )

        # Wrap: prefix + text + suffix + [MASK]*n_gen + structural tail
        gen_ids = [self.mask_token_id] * self.n_gen_tokens
        tail_ids = self._build_tail_ids(n_tail)
        mask_ids = gen_ids + tail_ids
        text_encodings['input_ids'] = [
            prefix_ids + ids + suffix_ids + mask_ids
            for ids in text_encodings['input_ids']
        ]

        collated = self.tokenizer.pad(
            text_encodings,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return collated

    # ------------------------------------------------------------------
    # Attention mask construction
    # ------------------------------------------------------------------

    def _build_full_attention_mask(
        self,
        seq_len: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build a 4D full bidirectional attention mask (only masks padding)."""
        device = attention_mask.device
        batch_size = attention_mask.size(0)

        dtype = self.backbone.dtype if hasattr(self.backbone, 'dtype') else torch.bfloat16
        min_val = torch.finfo(dtype).min
        mask_4d = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)
        pad_mask = ~attention_mask.bool()
        mask_4d = mask_4d.masked_fill(pad_mask.unsqueeze(1).unsqueeze(1), min_val)  # key cols
        mask_4d = mask_4d.masked_fill(pad_mask.unsqueeze(1).unsqueeze(3), min_val)  # query rows

        return mask_4d

    # ------------------------------------------------------------------
    # Hidden state extraction
    # ------------------------------------------------------------------

    def _get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get last hidden states from Dream.

        Returns:
            Hidden states [batch_size, seq_len, hidden_size].
        """
        kwargs = dict(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        if self._flash_attn:
            # Flash Attention 2 doesn't support 4D masks — use 2D padding mask.
            # With is_causal=False (Dream default), this gives bidirectional attention.
            kwargs["attention_mask"] = attention_mask
        elif attention_mask_4d is not None:
            kwargs["attention_mask"] = attention_mask_4d
        else:
            kwargs["attention_mask"] = self._build_full_attention_mask(
                input_ids.size(1), attention_mask
            )

        outputs = self.backbone(**kwargs)

        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        raise RuntimeError(
            "Dream model did not return hidden_states. "
            "Ensure output_hidden_states=True is supported."
        )

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pool(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings into a single sequence embedding."""
        mask = attention_mask.unsqueeze(-1).float()

        if self.pooling == "mean":
            return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "weighted_mean":
            seq_len = token_embeddings.size(1)
            weights = torch.arange(1, seq_len + 1, device=token_embeddings.device, dtype=torch.float)
            weights = weights.unsqueeze(0).unsqueeze(-1) * mask
            return (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "last":
            # With left-padding, the last real token is always at position -1
            return token_embeddings[:, -1, :]
        elif self.pooling == "attention":
            attn_weights = self.attn_pool(token_embeddings).squeeze(-1)
            attn_weights = attn_weights.masked_fill(~attention_mask.bool(), float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)
            return (token_embeddings * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")


    # ------------------------------------------------------------------
    # Encoding mode 1: Clean
    # ------------------------------------------------------------------

    def encode_clean(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Clean encoding — Dream as bidirectional encoder."""
        hidden_states = self._get_hidden_states(input_ids, attention_mask)
        return self._pool(hidden_states.float(), attention_mask)

    # ------------------------------------------------------------------
    # Encoding mode 2: PromptReps
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_promptreps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encode_type: str = 'all',
        is_query: bool = False,
        content_token_ids: List = None,
    ) -> Dict[str, torch.Tensor]:
        """PromptReps encoding — Dream version.

        Sequence layout: [prefix][text][suffix]["][MASK×n_gen][EOS]

        All n_gen [MASK] tokens are decoded step-by-step. At the step each token
        transitions from MASK → decoded, we save:
          - its hidden state (repr_hidden, for ColBERT-style multi-vector dense)
          - its top-128 sparse logits (sparse_indices, sparse_values)

        all_steps output shapes:
          repr_hidden    [B, n_gen, H]      dense per MASK token at decode step
          sparse_indices [B, n_gen, 128]
          sparse_values  [B, n_gen, 128]
        """
        need_dense = encode_type in ('all', 'dense')
        need_sparse = encode_type in ('all', 'sparse')
        need_all = encode_type == 'all_steps'

        device = input_ids.device
        batch_size = input_ids.size(0)
        n_gen = self.n_gen_tokens
        num_steps = self.num_denoise_steps
        seq_len = input_ids.size(1)

        curr_ids = input_ids.clone()

        # With left padding, MASK block is at [g_start : g_start+n_gen+n_tail],
        # where the first n_gen positions are representation tokens and the last
        # n_tail positions absorb structural closing tokens (" / EOS).
        n_tail = self._n_tail if n_gen > 0 else 0
        n_total = n_gen + n_tail  # total MASK block size
        L = seq_len
        g_start = L - n_total  # constant for all items in batch
        quot_pos = g_start - 1  # position of the " token

        fwd_mask = self._build_full_attention_mask(seq_len, attention_mask)

        all_hidden = []
        sparse_logits = None

        eps = 1e-3
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

        def _repr_dense(hidden):
            """[B, H] — dense embedding from MASK positions (mean if multiple)."""
            if n_gen == 0:
                # Fallback: use last real token if no MASK tokens
                return hidden[:, g_start - 1, :]
            return hidden[:, g_start:g_start + n_gen, :].mean(dim=1)

        def _repr_sparse(logits):
            """[B, V] — max-pooled logits over gen positions (step-0, all-MASK context)."""
            if n_gen == 0:
                return torch.zeros(batch_size, logits.shape[-1], device=device)
            return logits[:, g_start:g_start + n_gen, :].max(dim=1).values

        def _save_tok_sparse(tok_log_1v, i, tok_pos):
            """Save per-position sparse for example i at token position tok_pos.

            Applies log(1+relu) then topk(128) independently per position,
            matching PromptReps' per-position sparse approach (aggregated via
            sum at retrieval time rather than global max-pool at encode time).
            """
            if content_token_ids is not None:
                tok_log_1v = filter_sparse(tok_log_1v, [content_token_ids[i]],
                                           exclude_ids=[self.mask_token_id])
            else:
                tok_log_1v = tok_log_1v.clone()
                tok_log_1v[:, self.mask_token_id] = 0.0
            tok_log_1v = torch.log(1 + torch.relu(tok_log_1v)).squeeze(0)
            vals, idxs = tok_log_1v.topk(128)
            vals = (vals * 100).int().float()   # PromptReps quantization step
            repr_sparse_indices[i, tok_pos] = idxs
            repr_sparse_values[i, tok_pos] = vals

        # all_steps per-token tracking tensors
        if need_all and n_gen > 0:
            repr_hidden_all = torch.zeros(batch_size, n_gen, self.hidden_size, device=device)
            repr_saved = torch.zeros(batch_size, n_gen, dtype=torch.bool, device=device)
            repr_sparse_indices = torch.zeros(batch_size, n_gen, 128, dtype=torch.long, device=device)
            repr_sparse_values = torch.zeros(batch_size, n_gen, 128, device=device)

        for step in range(num_steps):
            outputs = self.backbone(
                input_ids=curr_ids, attention_mask=fwd_mask,
                output_hidden_states=(need_dense or need_all), return_dict=True,
            )

            if need_dense or need_all:
                h = outputs.hidden_states[-1].float()
                if need_dense:
                    all_hidden.append(_repr_dense(h))

            if need_sparse and step == 0:
                sparse_logits = _repr_sparse(outputs.logits)

            if n_total > 0:
                t = timesteps[step]
                s = timesteps[step + 1]
                for i in range(batch_size):
                    # Denoise the full MASK block (n_gen repr + n_tail closing)
                    gen_ids = curr_ids[i, g_start:g_start + n_total]
                    mask_pos = (gen_ids == self.mask_token_id)
                    num_masked = mask_pos.sum().item()
                    if num_masked == 0:
                        continue
                    mask_logits = outputs.logits[i, g_start:g_start + n_total][mask_pos].clone()
                    if not torch.isfinite(mask_logits).all():
                        continue
                    confidence, x0 = self._sample_with_confidence(mask_logits)
                    if step == num_steps - 1:
                        num_transfer = num_masked
                    else:
                        num_transfer = max(1, int(num_masked * (1 - s / t)))
                    num_transfer = min(num_transfer, num_masked)
                    _, transfer_idx = torch.topk(confidence, num_transfer)
                    masked_abs = torch.where(mask_pos)[0]

                    # Save hidden state + sparse only for repr positions (0..n_gen-1)
                    if need_all:
                        for j_item in range(transfer_idx.shape[0]):
                            tok_pos = masked_abs[transfer_idx[j_item]].item()
                            if tok_pos < n_gen and not repr_saved[i, tok_pos]:
                                repr_hidden_all[i, tok_pos] = h[i, g_start + tok_pos, :]
                                _save_tok_sparse(
                                    outputs.logits[i, g_start + tok_pos, :].float().unsqueeze(0),
                                    i, tok_pos,
                                )
                                repr_saved[i, tok_pos] = True

                    curr_ids[i, g_start + masked_abs[transfer_idx]] = x0[transfer_idx]

            # Last step: save any repr tokens not yet saved (skipped due to NaN logits)
            if need_all and step == num_steps - 1:
                if n_gen > 0:
                    for i in range(batch_size):
                        for tok_pos in range(n_gen):  # only repr positions, not tail
                            if not repr_saved[i, tok_pos]:
                                repr_hidden_all[i, tok_pos] = h[i, g_start + tok_pos, :]
                                _save_tok_sparse(
                                    outputs.logits[i, g_start + tok_pos, :].float().unsqueeze(0),
                                    i, tok_pos,
                                )

        result = {}

        if need_dense:
            if len(all_hidden) == 1:
                dense = torch.nan_to_num(all_hidden[0], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                stacked = torch.stack(all_hidden, dim=1)  # [B, num_passes, H]
                h0 = torch.nan_to_num(stacked[:, :1, :], nan=0.0, posinf=0.0, neginf=0.0)
                stacked = torch.where(torch.isfinite(stacked), stacked, h0.expand_as(stacked))
                dense = stacked.mean(dim=1)
            result['dense'] = dense

        if need_sparse and sparse_logits is not None:
            sparse_logits = filter_sparse(sparse_logits, content_token_ids,
                                          exclude_ids=[self.mask_token_id])
            result['sparse'] = torch.log(1 + torch.relu(sparse_logits))

        if need_all:
            if n_gen > 0:
                # Truncate at the first stop token in the final decoded gen sequence.
                # Stop tokens: EOS, or any token whose string contains '"' (closing quote).
                # Once committed, a stop token stays fixed in curr_ids for all subsequent
                # steps — positions at and after it carry no semantic content.
                # Exception: if the very first position is a stop token (model confused),
                # keep everything as a fallback rather than producing a zero embedding.
                # Truncate at any special token (EOS variants: <|endoftext|>=151643,
                # <|im_end|>=151645, etc.). Use all_special_ids rather than eos_token_id
                # alone since Qwen2 often emits 151643 while eos_token_id=151645.
                # The closing '"' is a regular vocab token and is left in repr_hidden
                # as minor noise rather than risking false truncation on word tokens
                # that happen to contain '"' in their string representation.
                stop_ids = set(self.tokenizer.all_special_ids) - {self.mask_token_id}
                decoded_ids = curr_ids[:, g_start:g_start + n_gen].clone()  # [B, n_gen] — repr positions only
                n_truncated = 0
                for i in range(batch_size):
                    final_gen = curr_ids[i, g_start:g_start + n_gen]  # repr positions only
                    cutoff = n_gen  # sentinel: no stop token found
                    for k in range(n_gen):
                        if final_gen[k].item() in stop_ids:
                            cutoff = k
                            break
                    if cutoff < n_gen:
                        n_truncated += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            toks = [self.tokenizer.decode([final_gen[k].item()]) for k in range(n_gen)]
                            logger.debug(
                                f"Stop-token truncation at pos {cutoff}: {toks} "
                                f"(stop_id={final_gen[cutoff].item()})"
                            )
                    if cutoff > 0:  # cutoff==0 → model confused, keep all as fallback
                        repr_hidden_all[i, cutoff:] = 0.0
                        repr_sparse_values[i, cutoff:] = 0.0
                if n_truncated > 0:
                    logger.debug(f"Stop-token truncation hit {n_truncated}/{batch_size} examples in this batch")
                # Per-position masking: zero out positions that decoded to quote-containing
                # tokens (structural noise — model generating closing sequence early).
                # Non-contiguous: semantic positions after a structural one are kept.
                if self._structural_token_ids:
                    for i in range(batch_size):
                        for k in range(n_gen):
                            tok = curr_ids[i, g_start + k].item()
                            if tok in self._structural_token_ids:
                                repr_hidden_all[i, k] = 0.0
                                repr_sparse_values[i, k] = 0.0
                repr_hidden_all = torch.nan_to_num(repr_hidden_all, nan=0.0, posinf=0.0, neginf=0.0)
                result['repr_hidden'] = repr_hidden_all          # [B, n_gen, H]
                result['sparse_indices'] = repr_sparse_indices   # [B, n_gen, 128]
                result['sparse_values'] = repr_sparse_values     # [B, n_gen, 128]
                result['decoded_ids'] = decoded_ids              # [B, n_gen] raw decoded token IDs (before truncation)
        return result

    # ------------------------------------------------------------------
    # Encoding mode 3: Multi-vector / ColBERT-style
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_multivec(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encode_type: str = 'all',
        is_query: bool = False,
        content_token_ids: List = None,
    ) -> Dict[str, torch.Tensor]:
        """Multi-vector encoding — ColBERT-style.

        Single forward pass with num_repr_tokens [MASK] tokens.
        Returns K hidden states per document instead of mean-pooling.

        Returns:
            'dense': [B, K, D]  — K vectors per example
            'sparse': [B, V]    — max-pooled logits across K positions
        """
        need_dense = encode_type in ('all', 'dense')
        need_sparse = encode_type in ('all', 'sparse')

        device = input_ids.device
        batch_size = input_ids.size(0)
        n_repr = self.num_repr_tokens
        seq_len = input_ids.size(1)

        fwd_mask = self._build_full_attention_mask(seq_len, attention_mask)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=fwd_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        result = {}

        # With left padding, repr tokens are at [-n_repr:]
        r_start = seq_len - n_repr
        r_end = r_start + n_repr
        if need_dense:
            hidden = outputs.last_hidden_state.float()
            result['dense'] = hidden[:, r_start:r_end, :]  # [B, K, D]

        if need_sparse:
            sparse = outputs.logits[:, r_start:r_end, :].max(dim=1).values
            sparse = filter_sparse(sparse, content_token_ids, exclude_ids=[self.mask_token_id])
            result['sparse'] = torch.log(1 + torch.relu(sparse))

        return result

    def _sample_with_confidence(self, logits, temperature=0.0, alg='entropy'):
        """Sample tokens and compute confidence scores.

        Args:
            logits: [N, V] logits at masked positions
            temperature: sampling temperature (0 = greedy)
            alg: confidence scoring algorithm
                 'maskgit_plus' = top-1 prob
                 'topk_margin' = top1 - top2
                 'entropy' = negative entropy
        Returns:
            confidence: [N] scores
            x0: [N] predicted token IDs
        """
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            scaled_logits = logits

        probs = torch.softmax(scaled_logits, dim=-1)

        if temperature > 0:
            x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        else:
            confidence, x0 = probs.max(dim=-1)

        if alg == 'topk_margin':
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
        elif alg == 'entropy':
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            confidence = torch.sum(probs * log_probs, dim=-1)  # neg entropy
        # else: 'maskgit_plus' uses default top-1 prob confidence

        return confidence, x0

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoding_mode: Literal["clean", "promptreps"] = "clean",
        encode_type: str = 'all',
        is_query: bool = False,
        content_token_ids: List = None,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            pad_id = getattr(self.tokenizer, 'pad_token_id', None) or 0
            attention_mask = (input_ids != pad_id).long()

        if encoding_mode == "promptreps":
            result = self.encode_promptreps(
                input_ids, attention_mask, encode_type=encode_type, is_query=is_query,
                content_token_ids=content_token_ids,
            )
            if 'dense' in result:
                if self.normalize:
                    result['dense'] = F.normalize(result['dense'], p=2, dim=-1)
                result['embeddings'] = result['dense']
            if 'repr_hidden' in result and self.normalize:
                result['repr_hidden'] = F.normalize(result['repr_hidden'], p=2, dim=-1)  # [B, n_gen, H]
            return result

        if encoding_mode == "multivec":
            result = self.encode_multivec(
                input_ids, attention_mask, encode_type=encode_type, is_query=is_query,
                content_token_ids=content_token_ids,
            )
            if 'dense' in result and self.normalize:
                result['dense'] = F.normalize(result['dense'], p=2, dim=-1)
            if 'dense' in result:
                result['embeddings'] = result['dense']
            return result

        if encoding_mode == "clean":
            embeddings = self.encode_clean(input_ids, attention_mask)
        else:
            raise ValueError(f"Unknown encoding mode: {encoding_mode}")

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return {'embeddings': embeddings}

    # ------------------------------------------------------------------
    # High-level encode API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 8,
        encoding_mode: Literal["clean", "promptreps", "multivec"] = "clean",
        is_query: bool = True,
        show_progress: bool = True,
        encode_type: str = 'all',
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        all_embeddings = []
        all_sparse = []
        accum = {}   # for all_steps keys: repr_hidden, sparse_indices, sparse_values
        device = next(self.backbone.parameters()).device

        for i in range(0, len(texts), batch_size):
            if show_progress and i % (batch_size * 10) == 0:
                logger.info(f"Encoding {i}/{len(texts)}...")

            batch_texts = texts[i:i + batch_size]

            if encoding_mode in ("promptreps", "multivec", "multivec_diffusion"):
                encoded = self._tokenize_promptreps(batch_texts, is_query)
            else:
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Content token IDs for sparse filtering (stopwords removed)
            batch_content_ids = None
            needs_sparse = encode_type in ('all', 'sparse', 'all_steps')
            if encoding_mode in ("promptreps", "multivec", "multivec_diffusion") and needs_sparse:
                batch_content_ids = get_content_token_ids(batch_texts, self.tokenizer)

            outputs = self.forward(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                encoding_mode=encoding_mode,
                encode_type=encode_type,
                is_query=is_query,
                content_token_ids=batch_content_ids,
            )
            if 'embeddings' in outputs:
                all_embeddings.append(outputs['embeddings'].cpu())
            if 'sparse' in outputs:
                all_sparse.append(outputs['sparse'].cpu())
            # all_steps keys — pass through directly
            for key in ('repr_hidden', 'sparse_indices', 'sparse_values'):
                if key in outputs:
                    accum.setdefault(key, []).append(outputs[key].cpu())

        result = {}
        if all_embeddings:
            result['embeddings'] = torch.cat(all_embeddings, dim=0)
            result['dense'] = result['embeddings']
        if all_sparse:
            result['sparse'] = torch.cat(all_sparse, dim=0)
        for key, batches in accum.items():
            result[key] = torch.cat(batches, dim=0)
        return result
