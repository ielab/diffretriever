"""
DiffRetriever — LLaDA backbone (zero-shot).

Uses LLaDA (a from-scratch discrete diffusion LM) as the backbone for
masked-position prediction retrieval (see paper §3.2).  The model
appends K masked positions after the retrieval prompt and reads K
hidden states + K logit vectors from a single bidirectional forward pass.

Supports the full LLaDA family:
  - LLaDA v1   (GSAI-ML/LLaDA-8B-Instruct)   — custom LLaDA arch (vocab 126464), masked diffusion
  - LLaDA v1.5 (GSAI-ML/LLaDA-1.5)            — same arch as v1 (vocab 126464)
  - LLaDA v2   (inclusionAI/LLaDA2.0-mini)    — custom MoE arch (vocab ~156K), block diffusion

Encoding strategies:
1. PromptReps encoding: wraps text with a prompt template, appends [MASK] token(s),
   denoises them, uses hidden states at [MASK] positions as embeddings.
2. Block-interactive encoding (v2 only): block-causal progressive embeddings.
3. Clean encoding: single forward pass, all tokens visible (bidirectional).
4. Block-denoising encoding: full block [MASK] denoising (v2 only).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Literal, Tuple
from pathlib import Path
import logging

_NUM_WORDS = ['one', 'two', 'three', 'four', 'five',
              'six', 'seven', 'eight', 'nine', 'ten']

from .block_schedule import BlockSchedule
from .sparse_utils import get_content_token_ids, filter_sparse

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 156895  # LLaDA 2's [MASK] token id (<|mask|>)

# Known mask token IDs per model family
_KNOWN_MASK_IDS = {
    'GSAI-ML/LLaDA-8B-Instruct': 126336,      # LLaDA v1   (vocab 126464)
    'GSAI-ML/LLaDA-8B-Base': 126336,           # LLaDA v1   (vocab 126464)
    'GSAI-ML/LLaDA-1.5': 126336,              # LLaDA v1.5 (vocab 126464, same arch as v1)
    'inclusionAI/LLaDA2.0-mini': 156895,       # LLaDA v2   (custom MoE)
}


class LLaDA2Retriever(nn.Module):
    """
    Dense retriever built on LLaDA's masked diffusion process.

    Supports both LLaDA v1 (GSAI-ML/LLaDA-8B-Instruct) and
    LLaDA v2 (inclusionAI/LLaDA2.0-mini). The mask token ID is
    auto-detected from the tokenizer.

    Exploits block-causal attention to extract progressive embeddings
    from a single forward pass. Supports four encoding modes:
    - clean: standard bidirectional encoding
    - block_interactive: block-causal progressive embeddings (training + inference)
    - promptreps: LLaDA PromptReps — prompt + [MASK] → denoise → embedding
    - block_denoising: full block [MASK] denoising (zero-shot inference only)

    Embeddings are the raw hidden states from the backbone (no projection).
    """

    def __init__(
        self,
        model_name: str = "inclusionAI/LLaDA2.0-mini",
        max_length: int = 512,
        pooling: Literal["mean", "weighted_mean", "last", "attention"] = "mean",
        normalize: bool = True,
        block_schedule: Optional[BlockSchedule] = None,
        block_aggregation: Literal["last", "mean", "weighted_mean", "ema", "attention"] = "ema",
        freeze_backbone: bool = False,
        mask_token_id: int = None,
        num_repr_tokens: int = None,
        num_denoise_steps: int = 1,
        query_prompt: str = "",
        passage_prompt: str = "",
        use_quotation_token: bool = True,
        n_gen_tokens: int = 0,
        filter_structural: bool = False,
        attn_implementation: str = "flash_attention_2",
    ):
        """
        Args:
            model_name: HuggingFace model name (LLaDA 2 model).
            max_length: Maximum sequence length.
            pooling: Pooling strategy for token -> sequence embeddings.
            normalize: Whether to L2-normalize output embeddings.
            block_schedule: Block diffusion schedule configuration.
            block_aggregation: How to aggregate progressive block embeddings.
            freeze_backbone: Whether to freeze LLaDA 2 backbone parameters.
            mask_token_id: LLaDA 2's mask token id (default: 156895).
            num_repr_tokens: Number of [MASK] representation tokens for promptreps mode.
            num_denoise_steps: Denoising steps for promptreps (1=single pass, >1=iterative).
            query_prompt: Path to YAML prompt file for queries (chat template applied).
            passage_prompt: Path to YAML prompt file for passages (chat template applied).
            use_quotation_token: If True, extract hidden state of the " token (one before
                [MASK]) instead of [MASK] itself. Only applies when n_repr==1.
            n_gen_tokens: Number of [MASK] generation tokens (K). Controls both the
                prompt text ("Use K words") and the number of MASK tokens appended.
                An additional 3 tail MASK tokens are always appended to absorb structural
                closing tokens, so the first K positions decode to genuine content.
        """
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self.block_aggregation = block_aggregation
        self.num_denoise_steps = num_denoise_steps
        self.use_quotation_token = use_quotation_token
        self.n_gen_tokens = n_gen_tokens
        self._n_tail = 3  # extra MASK tokens appended to absorb structural closing tokens
        self.filter_structural = filter_structural

        # Block schedule
        self.block_schedule = block_schedule or BlockSchedule()
        # Default repr tokens = one full block (matches LLaDA2's block diffusion unit)
        self.num_repr_tokens = num_repr_tokens if num_repr_tokens is not None else self.block_schedule.block_length

        # Load LLaDA
        logger.info(f"Loading LLaDA from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self._structural_token_ids = self._compute_structural_ids() if filter_structural else frozenset()

        # Ensure pad_token is set (LLaMA-3 based models often lack one)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token (id={self.tokenizer.pad_token_id})")

        # Auto-detect mask token ID from tokenizer
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        else:
            self.mask_token_id = self._detect_mask_token(self.tokenizer, model_name)
        logger.info(f"Mask token ID: {self.mask_token_id}")
        try:
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                attn_implementation=attn_implementation,
            )
            logger.info(f"Using {attn_implementation}")
        except (ValueError, ImportError):
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
            logger.info(f"{attn_implementation} not available, using default attention")

        # Verify mask_token_id is within model's embedding table
        emb_size = self.backbone.get_input_embeddings().weight.shape[0]
        if self.mask_token_id >= emb_size:
            logger.warning(
                f"mask_token_id={self.mask_token_id} >= embedding size {emb_size}! "
                f"Resizing model embeddings to {self.mask_token_id + 1}."
            )
            self.backbone.resize_token_embeddings(self.mask_token_id + 1)
        logger.info(f"Model vocab/embedding size: {self.backbone.get_input_embeddings().weight.shape[0]}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("LLaDA backbone frozen")

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

        # Determine if we need 4D attention masks.
        # LLaDA-2 (inclusionAI) without flash attention uses a standard HF causal model
        # that needs a 4D mask to override causal attention → bidirectional.
        # LLaDA-1/1.5 (GSAI-ML) have custom modeling code that handles bidirectional
        # attention internally — they expect a 2D mask.
        _is_llada2 = 'inclusionAI' in model_name or 'LLaDA2' in model_name
        self._use_4d_mask = _is_llada2 and not self._flash_attn
        if self._use_4d_mask:
            logger.info("Using 4D attention masks (LLaDA2 without flash attention)")
        else:
            logger.info("Using 2D attention masks (model handles bidirectional internally)")

        # Determine hidden size (= embedding dim, no projection)
        self.hidden_size = getattr(
            self.backbone.config, 'hidden_size',
            getattr(self.backbone.config, 'd_model', 2048)
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

    @staticmethod
    def _detect_mask_token(tokenizer, model_name: str) -> int:
        """Detect the LLaDA diffusion mask token ID from the tokenizer.

        Strategy:
        1. Fall back to _KNOWN_MASK_IDS table (most reliable for known models)
        2. Check tokenizer.mask_token_id (set by some models)
        3. Try exact diffusion mask token strings (NOT fuzzy search —
           GLM's [gMASK] is NOT the diffusion mask)
        4. Search added_tokens for exact match on '<|mask|>' or '[MASK]'
        """
        # 1. Known table (most reliable for known models)
        if model_name in _KNOWN_MASK_IDS:
            logger.info(f"Mask token from _KNOWN_MASK_IDS[{model_name!r}]: {_KNOWN_MASK_IDS[model_name]}")
            return _KNOWN_MASK_IDS[model_name]

        # 2. Some tokenizers set mask_token directly
        if getattr(tokenizer, 'mask_token_id', None) is not None:
            logger.info(f"Mask token from tokenizer.mask_token_id: {tokenizer.mask_token_id}")
            return tokenizer.mask_token_id

        # 3. Try exact diffusion mask token strings
        for cand in ['<|mask|>', '[MASK]', '<mask>']:
            tid = tokenizer.convert_tokens_to_ids(cand)
            if tid is not None and tid != tokenizer.unk_token_id and tid != 0:
                logger.info(f"Mask token '{cand}' → {tid}")
                return tid

        # 4. Search added_tokens for exact match only (not fuzzy —
        #    e.g. [gMASK] is GLM's generative mask, NOT diffusion mask)
        added = getattr(tokenizer, 'added_tokens_encoder', {})
        for exact in ['<|mask|>', '[MASK]', '<mask>']:
            if exact in added:
                logger.info(f"Mask token from added_tokens: '{exact}' → {added[exact]}")
                return added[exact]

        raise ValueError(
            f"Cannot auto-detect diffusion mask token for {model_name}. "
            f"Added tokens: {list(added.keys())[:20]}. "
            f"Pass mask_token_id explicitly."
        )

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

    def _compute_structural_ids(self) -> frozenset:
        """Token IDs whose decoded text contains '"' — structural closing tokens."""
        structural = set()
        for tok_str in self.tokenizer.get_vocab():
            decoded = self.tokenizer.convert_tokens_to_string([tok_str])
            if '"' in decoded:
                structural.add(self.tokenizer.convert_tokens_to_ids(tok_str))
        return frozenset(structural)

    def _exact_token_id(self, token: str) -> Optional[int]:
        tok_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, 'unk_token_id', None)
        if tok_id is not None and tok_id >= 0 and tok_id != unk_id:
            return tok_id
        return None

    def _single_token_text_id(self, text: str) -> Optional[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _build_tail_ids(self, n_tail: int) -> List[int]:
        if n_tail <= 0:
            return []
        quote_id = self._single_token_text_id('"')
        eot_id = self._exact_token_id('<|eot_id|>')
        if eot_id is None:
            eot_id = 126348
        eos_id = self.tokenizer.eos_token_id
        tail_ids: List[int] = []
        if quote_id is not None:
            tail_ids.append(quote_id)
        if eot_id is not None and eot_id != eos_id:
            tail_ids.append(eot_id)
        while len(tail_ids) < n_tail:
            tail_ids.append(eos_id)
        return tail_ids[:n_tail]

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

    def _tokenize_promptreps(
        self, texts: List[str], is_query: bool,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts with prefix + text + suffix + [MASK] tokens.

        Matches the original PromptReps token-level wrapping, then appends
        num_repr_tokens [MASK] tokens at the end.
        """
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
    # Hidden state extraction
    # ------------------------------------------------------------------

    def _get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get last hidden states from LLaDA 2.

        Returns:
            Hidden states [batch_size, seq_len, hidden_size].
        """
        kwargs = dict(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        if attention_mask_4d is not None:
            # Explicit 4D mask override (e.g., block-causal) — use as-is.
            # Note: this will bypass Flash Attention (falls back to eager).
            kwargs["attention_mask"] = attention_mask_4d
        elif self._flash_attn:
            # Flash Attention 2 doesn't support 4D masks — use 2D padding mask.
            # With is_causal=False (LLaDA2 default), this gives bidirectional attention.
            kwargs["attention_mask"] = attention_mask
        else:
            # Non-flash: use 4D mask to override LLaDA2's internal causal mask.
            kwargs["attention_mask"] = self._build_full_attention_mask(
                input_ids.size(1), attention_mask
            )

        outputs = self.backbone(**kwargs)

        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        raise RuntimeError(
            "LLaDA 2 model did not return hidden_states. "
            "Ensure output_hidden_states=True is supported."
        )

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pool(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings into a single sequence embedding.

        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            Pooled embeddings [batch_size, hidden_size]
        """
        mask = attention_mask.unsqueeze(-1).float()

        if self.pooling == "mean":
            return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        elif self.pooling == "weighted_mean":
            seq_len = token_embeddings.size(1)
            weights = torch.arange(1, seq_len + 1, device=token_embeddings.device, dtype=torch.float)
            weights = weights.unsqueeze(0).unsqueeze(-1) * mask
            return (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)

        elif self.pooling == "last":
            # With left padding, the last real token is always at the final position.
            return token_embeddings[:, -1, :]

        elif self.pooling == "attention":
            attn_weights = self.attn_pool(token_embeddings).squeeze(-1)
            attn_weights = attn_weights.masked_fill(~attention_mask.bool(), float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)
            return (token_embeddings * attn_weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    # ------------------------------------------------------------------
    # Block-causal mask construction
    # ------------------------------------------------------------------

    def _build_block_causal_mask(
        self,
        seq_len: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build a 4D block-causal attention mask.

        Within a block: full bidirectional attention.
        Across blocks: causal (block k can attend to blocks 0..k but not k+1..).
        """
        device = attention_mask.device
        block_len = self.block_schedule.block_length

        positions = torch.arange(seq_len, device=device)
        block_ids = positions // block_len

        dtype = self.backbone.dtype if hasattr(self.backbone, 'dtype') else torch.bfloat16
        min_val = torch.finfo(dtype).min
        causal_mask = block_ids.unsqueeze(0) <= block_ids.unsqueeze(1)
        mask_2d = torch.where(causal_mask, torch.tensor(0.0, dtype=dtype), torch.tensor(min_val, dtype=dtype))

        mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).expand(
            attention_mask.size(0), 1, seq_len, seq_len
        ).clone()

        pad_mask = ~attention_mask.bool()
        mask_4d = mask_4d.masked_fill(pad_mask.unsqueeze(1).unsqueeze(1), min_val)  # key cols
        mask_4d = mask_4d.masked_fill(pad_mask.unsqueeze(1).unsqueeze(3), min_val)  # query rows

        return mask_4d

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
        # Mask both key columns and query rows for padding positions
        mask_4d = mask_4d.masked_fill(pad_mask.unsqueeze(1).unsqueeze(1), min_val)  # key cols
        mask_4d = mask_4d.masked_fill(pad_mask.unsqueeze(1).unsqueeze(3), min_val)  # query rows

        return mask_4d

    # ------------------------------------------------------------------
    # Block embedding aggregation
    # ------------------------------------------------------------------

    def _aggregate_block_embeddings(
        self, block_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate progressive block embeddings into a final embedding."""
        if len(block_embeddings) == 1:
            return block_embeddings[0]

        if self.block_aggregation == "last":
            return block_embeddings[-1]

        stacked = torch.stack(block_embeddings, dim=1)  # [B, K, dim]

        if self.block_aggregation == "mean":
            return stacked.mean(dim=1)

        elif self.block_aggregation == "weighted_mean":
            K = stacked.size(1)
            weights = torch.arange(1, K + 1, device=stacked.device, dtype=torch.float)
            weights = weights / weights.sum()
            weights = weights.unsqueeze(0).unsqueeze(-1)
            return (stacked * weights).sum(dim=1)

        elif self.block_aggregation == "ema":
            decay = torch.sigmoid(self.ema_decay_logit)
            K = stacked.size(1)
            ema_weights = torch.zeros(K, device=stacked.device)
            for k in range(K):
                ema_weights[k] = (1 - decay) * decay ** (K - 1 - k)
            ema_weights = ema_weights / ema_weights.sum()
            ema_weights = ema_weights.unsqueeze(0).unsqueeze(-1)
            return (stacked * ema_weights).sum(dim=1)

        elif self.block_aggregation == "attention":
            query = self.block_query.expand(stacked.size(0), -1, -1)
            out, _ = self.block_attn(query, stacked, stacked)
            return out.squeeze(1)

        else:
            raise ValueError(f"Unknown block aggregation: {self.block_aggregation}")

    # ------------------------------------------------------------------
    # Encoding mode 1: Clean
    # ------------------------------------------------------------------

    def encode_clean(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Clean encoding — LLaDA 2 as bidirectional encoder."""
        hidden_states = self._get_hidden_states(input_ids, attention_mask)
        return self._pool(hidden_states.float(), attention_mask)

    # ------------------------------------------------------------------
    # Encoding mode 2: Block-interactive (main innovation)
    # ------------------------------------------------------------------

    def encode_block_interactive(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Block-interactive encoding — progressive embeddings via block-causal attention."""
        if self._flash_attn:
            logger.warning(
                "Block-interactive mode requires 4D block-causal mask which is "
                "incompatible with Flash Attention 2. Falling back to 4D mask "
                "(Flash Attention will be bypassed for this call)."
            )
        seq_len = input_ids.size(1)

        block_causal_mask = self._build_block_causal_mask(seq_len, attention_mask)

        hidden_states = self._get_hidden_states(
            input_ids, attention_mask, attention_mask_4d=block_causal_mask
        )
        hidden_states = hidden_states.float()

        boundaries = self.block_schedule.get_block_boundaries(seq_len)
        block_embeddings = []

        for k, (_, end) in enumerate(boundaries):
            partial_emb = hidden_states[:, :end, :]
            partial_mask = attention_mask[:, :end]
            pooled = self._pool(partial_emb, partial_mask)
            block_embeddings.append(pooled)

        return self._aggregate_block_embeddings(block_embeddings)


    # ------------------------------------------------------------------
    # Confidence-based token sampling (for gen token decoding)
    # ------------------------------------------------------------------

    def _sample_with_confidence(self, logits, temperature=0.0, alg='entropy'):
        """Sample tokens and compute confidence scores.

        Args:
            logits: [N, V] logits at masked positions
            temperature: sampling temperature (0 = greedy)
            alg: confidence scoring — 'entropy' (neg entropy), 'topk_margin', 'maskgit_plus'
        Returns:
            confidence: [N] scores, x0: [N] predicted token IDs
        """
        scaled_logits = logits / temperature if temperature > 0 else logits
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
            log_probs = torch.log(probs + 1e-10)
            confidence = torch.sum(probs * log_probs, dim=-1)  # neg entropy

        return confidence, x0

    # ------------------------------------------------------------------
    # Encoding mode 3: PromptReps (LLaDA 2 version)
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
        """PromptReps encoding — LLaDA 2 version.

        Sequence layout: [prefix][text][suffix]["][MASK×n_gen]

        All n_gen [MASK] tokens are decoded step-by-step. At the step each token
        transitions from MASK → decoded, we save:
          - its hidden state (repr_hidden, for ColBERT-style multi-vector dense)
          - its top-128 sparse logits (sparse_indices, sparse_values)

        The quotation " token hidden state is saved at the FINAL forward pass
        (sees fully decoded context).

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
        # Left padding: layout is [...prefix text suffix " MASK×K EOS].
        # MASKs are at positions g_start:g_start+K, EOS is at the last position.
        # MASK block layout: [g_start : g_start+n_gen+n_tail]
        # First n_gen positions = repr tokens; last n_tail absorb structural closing.
        n_tail = self._n_tail if n_gen > 0 else 0
        n_total = n_gen + n_tail
        L = seq_len

        # LLaDA models with trust_remote_code handle bidirectional attention
        # internally — always pass 2D mask. Only use 4D for LLaDA2 without flash
        # attention (needs to override its internal causal mask).
        if self._use_4d_mask:
            fwd_mask = self._build_full_attention_mask(seq_len, attention_mask)
        else:
            fwd_mask = attention_mask

        all_hidden = []
        sparse_logits = None

        eps = 1e-3
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

        # g_start: first MASK position (accounts for trailing EOS)
        _g_start = L - n_total

        def _repr_dense(hidden):
            """[B, H] — dense embedding from MASK positions (mean if multiple)."""
            if n_gen == 0:
                # Fallback: use last real token if no MASK tokens
                return hidden[:, _g_start - 1, :]
            return hidden[:, _g_start:_g_start + n_gen, :].mean(dim=1)

        def _repr_sparse(logits):
            """[B, V] — max-pooled logits over gen positions (step-0, all-MASK context)."""
            if n_gen == 0:
                return torch.zeros(batch_size, logits.shape[-1], device=device)
            return logits[:, _g_start:_g_start + n_gen, :].max(dim=1).values

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
                g_start = _g_start
                t = timesteps[step]
                s = timesteps[step + 1]
                for i in range(batch_size):
                    # Denoise full MASK block (n_gen repr + n_tail closing)
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
            if need_all and step == num_steps - 1 and n_gen > 0:
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
                # All special tokens (e.g. <|endoftext|>=151643, <|im_end|>=151645)
                # are stop signals — use all_special_ids rather than eos_token_id alone,
                # since Qwen2 eos_token_id=151645 but the model often emits 151643.
                stop_ids = set(self.tokenizer.all_special_ids) - {self.mask_token_id}
                for i in range(batch_size):
                    final_gen = curr_ids[i, _g_start:_g_start + n_gen]
                    cutoff = n_gen  # sentinel: no stop token found
                    for k in range(n_gen):
                        if final_gen[k].item() in stop_ids:
                            cutoff = k
                            break
                    if cutoff > 0:  # cutoff==0 → model confused, keep all as fallback
                        repr_hidden_all[i, cutoff:] = 0.0
                        repr_sparse_values[i, cutoff:] = 0.0
                # Per-position masking: zero out positions that decoded to quote-containing
                # tokens (structural noise — model generating closing sequence early).
                # Non-contiguous: semantic positions after a structural one are kept.
                if self._structural_token_ids:
                    for i in range(batch_size):
                        for k in range(n_gen):
                            tok = curr_ids[i, _g_start + k].item()
                            if tok in self._structural_token_ids:
                                repr_hidden_all[i, k] = 0.0
                                repr_sparse_values[i, k] = 0.0
                repr_hidden_all = torch.nan_to_num(repr_hidden_all, nan=0.0, posinf=0.0, neginf=0.0)
                result['repr_hidden'] = repr_hidden_all          # [B, n_gen, H]
                result['sparse_indices'] = repr_sparse_indices   # [B, n_gen, 128]
                result['sparse_values'] = repr_sparse_values     # [B, n_gen, 128]

        return result

    # ------------------------------------------------------------------
    # Encoding mode 5: Multi-vector denoising (ColBERT-style)
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
        """Multi-vector denoising encoding (ColBERT-style).

        Like encode_promptreps but keeps K vectors per document instead of
        mean-pooling them. Uses only the LAST denoising step's hidden states.

        Args:
            encode_type: 'all' (both), 'dense' (skip sparse), 'sparse' (skip dense).
            is_query: Whether encoding queries.
            content_token_ids: Per-example sets of valid content token IDs.

        Returns:
            Dict with:
                'dense': [B, K, D] — K vectors per example (from last step)
                'sparse': [B, V] — max-pooled logits across K positions
        """
        need_dense = encode_type in ('all', 'dense')
        need_sparse = encode_type in ('all', 'sparse')

        device = input_ids.device
        batch_size = input_ids.size(0)
        n_repr = self.num_repr_tokens
        num_steps = self.num_denoise_steps
        seq_len = input_ids.size(1)

        curr_ids = input_ids.clone()
        r_start = seq_len - n_repr

        if self._flash_attn:
            fwd_mask = attention_mask
        else:
            fwd_mask = self._build_full_attention_mask(seq_len, attention_mask)

        for step in range(num_steps):
            outputs = self.backbone(
                input_ids=curr_ids,
                attention_mask=fwd_mask,
                output_hidden_states=need_dense,
                return_dict=True,
            )

            # Only extract representations from the LAST step
            if step == num_steps - 1:
                result = {}

                if need_dense:
                    hidden = outputs.hidden_states[-1].float()
                    # Gather all K vectors per example: [B, K, D]
                    result['dense'] = hidden[:, r_start:r_start + n_repr, :]  # [B, K, D]

                if need_sparse:
                    logits = outputs.logits
                    # Max-pool logits across K positions → [B, V]
                    sparse = logits[:, r_start:r_start + n_repr, :].max(dim=1).values
                    sparse = filter_sparse(
                        sparse, content_token_ids,
                        exclude_ids=[self.mask_token_id],
                    )
                    sparse = torch.log(1 + torch.relu(sparse))
                    result['sparse'] = sparse

                return result

            # Denoise: replace [MASK] with predictions for next step
            pred = outputs.logits[:, r_start:r_start + n_repr, :].argmax(dim=-1)
            curr_ids[:, r_start:r_start + n_repr] = pred

    # ------------------------------------------------------------------
    # Encoding mode 4: Block-denoising (zero-shot inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_block_denoising(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Block-denoising encoding — actual denoising from [MASK] tokens.

        Non-differentiable (inference only).
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        prompt_len = input_ids.size(1)
        block_len = self.block_schedule.block_length
        num_steps = self.block_schedule.num_steps_per_block

        num_response_blocks = 1
        response_len = num_response_blocks * block_len

        response_ids = torch.full(
            (batch_size, response_len), self.mask_token_id,
            dtype=input_ids.dtype, device=device,
        )
        response_mask = torch.ones(
            (batch_size, response_len), dtype=attention_mask.dtype, device=device,
        )

        full_ids = torch.cat([input_ids, response_ids], dim=1)
        full_mask = torch.cat([attention_mask, response_mask], dim=1)

        block_embeddings = []

        for block_idx in range(num_response_blocks):
            block_start = prompt_len + block_idx * block_len
            block_end = block_start + block_len

            block_positions = list(range(block_start, block_end))
            masked_positions = set(block_positions)
            tokens_per_step = max(1, block_len // num_steps)

            for step in range(num_steps):
                if not masked_positions:
                    break

                if self._flash_attn:
                    fwd_mask = full_mask
                else:
                    fwd_mask = self._build_full_attention_mask(
                        full_ids.size(1), full_mask
                    )
                outputs = self.backbone(
                    input_ids=full_ids,
                    attention_mask=fwd_mask,
                    return_dict=True,
                )

                logits = outputs.logits

                masked_pos_list = sorted(masked_positions)
                if not masked_pos_list:
                    break

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
                    revealed_in_block = [p for p in block_positions if p not in masked_positions]
                    if revealed_in_block:
                        rev_tensor = torch.tensor(revealed_in_block, device=device)
                        rev_logits = logits[:, rev_tensor, :]
                        rev_tokens = rev_logits.argmax(dim=-1)
                        full_ids[:, rev_tensor] = rev_tokens

            visible_end = block_end
            hidden_states = self._get_hidden_states(
                full_ids[:, :visible_end],
                full_mask[:, :visible_end],
            )
            pooled = self._pool(hidden_states.float(), full_mask[:, :visible_end])
            block_embeddings.append(pooled)

        return self._aggregate_block_embeddings(block_embeddings)

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoding_mode: Literal["clean", "block_interactive", "promptreps", "block_denoising", "multivec"] = "clean",
        encode_type: str = 'all',
        is_query: bool = False,
        content_token_ids: List = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            encode_type: 'all' (both), 'dense' (skip sparse), 'sparse' (skip dense).
                Only applies to promptreps/multivec modes.
            is_query: Whether encoding queries (for sparse token filtering).
            content_token_ids: Per-example sets of valid content token IDs for sparse filtering.

        Returns:
            Dict with 'embeddings' and/or 'dense'/'sparse' keys.
        """
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
            if 'dense' in result:
                if self.normalize:
                    # Normalize each of the K vectors independently: [B, K, D]
                    result['dense'] = F.normalize(result['dense'], p=2, dim=-1)
                result['embeddings'] = result['dense']
            return result

        if encoding_mode == "clean":
            embeddings = self.encode_clean(input_ids, attention_mask)
        elif encoding_mode == "block_interactive":
            embeddings = self.encode_block_interactive(input_ids, attention_mask)
        elif encoding_mode == "block_denoising":
            embeddings = self.encode_block_denoising(input_ids, attention_mask)
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
        encoding_mode: Literal["clean", "block_interactive", "promptreps", "block_denoising", "multivec"] = "clean",
        is_query: bool = True,
        show_progress: bool = True,
        encode_type: str = 'all',
    ) -> Dict[str, torch.Tensor]:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings.
            batch_size: Batch size.
            encoding_mode: Encoding strategy.
            is_query: For promptreps, selects query vs passage prefix/suffix.
            show_progress: Log progress.
            encode_type: 'all' (both), 'dense' (skip sparse), 'sparse' (skip dense).

        Returns:
            Dict with 'embeddings'/'dense' and/or 'sparse' based on encode_type.
        """
        self.eval()
        all_embeddings = []
        all_sparse = []
        accum = {}   # for all_steps keys: repr_hidden, sparse_indices, sparse_values
        device = next(self.backbone.parameters()).device

        for i in range(0, len(texts), batch_size):
            if show_progress and i % (batch_size * 10) == 0:
                logger.info(f"Encoding {i}/{len(texts)}...")

            batch_texts = texts[i:i + batch_size]

            if encoding_mode in ("promptreps", "multivec"):
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
            if encoding_mode in ("promptreps", "multivec") and needs_sparse:
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

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_pretrained(self, save_path: str):
        """Save model components."""
        import json, os
        os.makedirs(save_path, exist_ok=True)

        self.backbone.save_pretrained(f"{save_path}/backbone")
        self.tokenizer.save_pretrained(f"{save_path}/backbone")

        # Save retriever-specific weights (aggregation layers only, no projection)
        retriever_state = {}
        if self.pooling == "attention":
            retriever_state['attn_pool'] = self.attn_pool.state_dict()
        if self.block_aggregation == "ema":
            retriever_state['ema_decay_logit'] = self.ema_decay_logit.data
        elif self.block_aggregation == "attention":
            retriever_state['block_attn'] = self.block_attn.state_dict()
            retriever_state['block_query'] = self.block_query.data

        if retriever_state:
            torch.save(retriever_state, f"{save_path}/retriever_head.pt")

        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'pooling': self.pooling,
            'normalize': self.normalize,
            'block_aggregation': self.block_aggregation,
            'mask_token_id': self.mask_token_id,
            'num_repr_tokens': self.num_repr_tokens,
            'num_denoise_steps': self.num_denoise_steps,
            'query_prefix': self.query_prefix,
            'query_suffix': self.query_suffix,
            'passage_prefix': self.passage_prefix,
            'passage_suffix': self.passage_suffix,
            'block_schedule': {
                'block_length': self.block_schedule.block_length,
                'num_steps_per_block': self.block_schedule.num_steps_per_block,
                'enable_t2t': self.block_schedule.enable_t2t,
            },
        }
        with open(f"{save_path}/retriever_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):
        """Load from saved checkpoint."""
        import json

        with open(f"{load_path}/retriever_config.json") as f:
            config = json.load(f)

        schedule_info = config.pop('block_schedule', {})
        block_schedule = BlockSchedule(**schedule_info)

        config.update(kwargs)
        config['model_name'] = f"{load_path}/backbone"
        config['block_schedule'] = block_schedule

        model = cls(**config)

        head_path = f"{load_path}/retriever_head.pt"
        try:
            retriever_state = torch.load(head_path, map_location='cpu', weights_only=True)
            if 'attn_pool' in retriever_state and hasattr(model, 'attn_pool'):
                model.attn_pool.load_state_dict(retriever_state['attn_pool'])
            if 'ema_decay_logit' in retriever_state and hasattr(model, 'ema_decay_logit'):
                model.ema_decay_logit.data = retriever_state['ema_decay_logit']
            if 'block_attn' in retriever_state and hasattr(model, 'block_attn'):
                model.block_attn.load_state_dict(retriever_state['block_attn'])
                model.block_query.data = retriever_state['block_query']
        except FileNotFoundError:
            pass

        return model
