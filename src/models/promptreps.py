"""
PromptReps Baseline — Faithful reproduction of ielab/PromptReps.

Reference: https://github.com/ielab/PromptReps
Paper: "Prompting Large Language Models to Generate Dense and Sparse
        Representations for Zero-Shot Document Retrieval" (EMNLP 2024)

Supports:
- Single-representation mode (num_pooled_tokens=0): one forward pass, hidden state
  at last token position = dense embedding, logits = sparse embedding.
- Multi-representation mode (num_pooled_tokens>0): autoregressive generation of
  multiple tokens (stops at closing quote), mean-pool hidden states (dense),
  max-pool logits (sparse).
- Hybrid dense+sparse retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from typing import Optional, List, Dict, Literal, Tuple
from pathlib import Path
import logging

from .sparse_utils import get_content_token_ids, filter_sparse

logger = logging.getLogger(__name__)


class PromptRepsRetriever(nn.Module):
    """
    AR PromptReps baseline retriever.

    Matches the original ielab/PromptReps implementation:
    - Prefix/suffix prompt format (loaded from files or strings)
    - Dense: last-layer hidden state at last token position
    - Sparse: next-token logits with log(1 + ReLU(x)) activation
    - Single-representation (K=1, num_pooled_tokens=0) or
      multi-representation (K>1, num_pooled_tokens>0, sequential AR decoding)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_length: int = 512,
        normalize: bool = False,
        num_pooled_tokens: int = 0,
        query_prompt: str = "",
        passage_prompt: str = "",
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.num_pooled_tokens = num_pooled_tokens

        logger.info(f"Loading AR model from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Always left-pad: last real token (the '"') is always at position -1
        self.tokenizer.padding_side = 'left'

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                attn_implementation=attn_implementation,
            )
            logger.info(f"Using {attn_implementation}")
        except (ValueError, ImportError):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
            logger.info(f"{attn_implementation} not available, using default attention")

        # Build prefix/suffix token IDs from YAML prompt files (chat template).
        q_yaml = self._load_yaml_prompt(query_prompt)
        p_yaml = self._load_yaml_prompt(passage_prompt)
        if q_yaml is None or p_yaml is None:
            raise ValueError(
                "--query_prompt and --passage_prompt must be paths to valid YAML files. "
                f"Got: query_prompt={query_prompt!r}, passage_prompt={passage_prompt!r}"
            )
        self._query_prefix_ids, self._query_suffix_ids = self._build_chat_prompt_ids(q_yaml)
        self._passage_prefix_ids, self._passage_suffix_ids = self._build_chat_prompt_ids(p_yaml)

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
          [prefix_ids] [text_ids] [suffix_ids]
        where the last token of suffix_ids is the quotation '"' from assistant_prefix.
        """
        system = yaml_dict.get('system', '')
        user_prefix = yaml_dict.get('user_prefix', '')
        user_suffix = yaml_dict.get('user_suffix', '')
        assistant_prefix = yaml_dict.get('assistant_prefix', '')

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

        logger.info(f"Chat prompt: prefix={len(prefix_ids)} tokens, suffix={len(suffix_ids)} tokens")
        logger.info(f"  user_suffix: {user_suffix!r}")
        logger.info(f"  assistant_prefix: {assistant_prefix!r}")
        return prefix_ids, suffix_ids

    def _tokenize_with_prefix_suffix(
        self,
        texts: List[str],
        is_query: bool,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts with prefix/suffix at token level (matching original)."""
        prefix_ids = self._query_prefix_ids if is_query else self._passage_prefix_ids
        suffix_ids = self._query_suffix_ids if is_query else self._passage_suffix_ids

        # Tokenize texts without special tokens, then wrap with prefix/suffix
        text_encodings = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length - len(prefix_ids) - len(suffix_ids),
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )

        # Prepend prefix, append suffix at token level
        text_encodings['input_ids'] = [
            prefix_ids + ids + suffix_ids for ids in text_encodings['input_ids']
        ]

        # Pad
        collated = self.tokenizer.pad(
            text_encodings,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return collated

    def encode_single_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encode_type: str = 'all',
        content_token_ids: List = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Single-representation PromptReps (num_pooled_tokens=0).

        Args:
            encode_type: 'all' (both), 'dense' (skip sparse), 'sparse' (skip dense).
            content_token_ids: Per-example sets of valid content token IDs
                (from sparse_utils.get_content_token_ids). Used for sparse filtering.

        Returns:
            (dense_emb, sparse_emb): either may be None based on encode_type.
        """
        need_dense = encode_type in ('all', 'dense')
        need_sparse = encode_type in ('all', 'sparse')

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_dense,
            return_dict=True,
        )

        # With left-padding, the last real token (closing ") is always at position -1
        dense = None
        if need_dense:
            dense = outputs.hidden_states[-1][:, -1, :]
            if self.normalize:
                dense = F.normalize(dense, p=2, dim=-1)

        sparse = None
        if need_sparse:
            sparse = outputs.logits[:, -1, :]
            sparse = filter_sparse(sparse, content_token_ids)
            sparse = torch.log(1 + torch.relu(sparse))

        return dense, sparse

    def encode_multi_token_all_steps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_topk: int = 128,
        texts: List[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Multi-representation all_steps mode: per-position repr_hidden + quotation_emb + sparse.

        Matches original PromptReps multi_reps layout:
          - repr_hidden[:, 0, :] = hidden state at closing '"' (quotation mark)
          - repr_hidden[:, k, :] = hidden state at (k-1)-th generated token (k >= 1)
          So K vectors come from 1 prompt pass + (K-1) generation steps.

        Output format (compatible with evaluate_sweep.py):
          quotation_emb  [B, H]          — same as repr_hidden[:, 0, :]
          repr_hidden    [B, K, H]        — ['"' hidden, word1 hidden, ..., word_{K-1} hidden]
          sparse_indices [B, K, topk]     — per-position top-k token indices
          sparse_values  [B, K, topk]     — per-position top-k log(1+relu(logit)) scores

        Sparse logit alignment:
          repr_logits[:, 0, :] = logits at '"' position, predicting 1st word
          repr_logits[:, k, :] = logits at (k-1)-th generated token, predicting k-th word
        """
        device = input_ids.device
        orig_batch_size = input_ids.size(0)
        batch_size = orig_batch_size
        K = self.num_pooled_tokens

        # Pass 0: full prompt → hidden at '"' + logits predicting 1st word.
        # Left-padded, so the last real token is always at position -1.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )
        quotation_emb = outputs.hidden_states[-1][:, -1, :].float().clone()  # [B, H]
        if self.normalize:
            quotation_emb = F.normalize(quotation_emb, p=2, dim=-1)

        hidden_size = quotation_emb.size(-1)
        vocab_size  = outputs.logits.size(-1)
        past_key_values = outputs.past_key_values

        repr_hidden  = torch.zeros(batch_size, K, hidden_size, device=device)
        repr_logits  = torch.zeros(batch_size, K, vocab_size,  device=device)
        generated_ids = torch.zeros(batch_size, K, dtype=torch.long, device=device)

        # Position 0: '"' hidden state and logits (from prompt pass)
        repr_hidden[:, 0, :] = quotation_emb
        repr_logits[:, 0, :] = outputs.logits[:, -1, :].float()

        # Generate K-1 tokens, placing their hidden states at positions 1..K-1
        if K > 1:
            curr_ids = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # [B, 1]
            curr_mask = torch.cat(
                [attention_mask, torch.ones(batch_size, 1, device=device)], dim=1
            )

            for step in range(K - 1):
                # Do NOT pass position_ids explicitly: HuggingFace derives the correct
                # position from attention_mask (cumsum - 1), giving the same value as
                # (prompt_lens + step) but avoiding Flash Attention 2's
                # _prepare_fa2_from_position_ids path, which crashes with
                # "max() on empty tensor" when position_ids is passed with use_cache=True.
                outputs = self.model(
                    input_ids=curr_ids,
                    attention_mask=curr_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values

                repr_hidden[:, step + 1, :] = outputs.hidden_states[-1][:, -1, :].float()
                repr_logits[:, step + 1, :] = outputs.logits[:, -1, :].float()
                generated_ids[:, step + 1] = curr_ids[:, 0]

                curr_ids  = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                curr_mask = torch.cat(
                    [curr_mask, torch.ones(batch_size, 1, device=device)], dim=1
                )

        # EOS truncation: zero positions at/after first stop token in generated tokens.
        # Stop tokens: EOS/special tokens AND the closing quote '"' (end-of-generation marker).
        # Position 0 ('"' mark) is never truncated. Only check positions 1..K-1.
        stop_ids = set(self.tokenizer.all_special_ids)
        # Add closing quote token(s) — signals model has finished its word generation.
        # These are the regular ASCII double-quote, not typographic quotes.
        for _qid in self.tokenizer.encode('"', add_special_tokens=False):
            stop_ids.add(_qid)
        for i in range(batch_size):
            for k in range(1, K):
                if generated_ids[i, k].item() in stop_ids:
                    repr_hidden[i, k:] = 0.0
                    repr_logits[i, k:] = 0.0
                    break

        # Content filtering: only keep tokens appearing in the source text (matching original)
        if texts is not None:
            content_token_ids = get_content_token_ids(texts, self.tokenizer)
            for k in range(K):
                repr_logits[:, k, :] = filter_sparse(repr_logits[:, k, :], content_token_ids)

        # Per-position sparse: log(1 + relu(logit)), then top-k
        logits_act = torch.log(1.0 + torch.relu(repr_logits))  # [B, K, V]
        sp_vals, sp_idxs = logits_act.topk(sparse_topk, dim=-1)

        return {
            'repr_hidden':    repr_hidden[:orig_batch_size].to(torch.bfloat16),
            'quotation_emb':  quotation_emb[:orig_batch_size].to(torch.bfloat16),
            'sparse_indices': sp_idxs[:orig_batch_size],
            'sparse_values':  sp_vals[:orig_batch_size].to(torch.bfloat16),
        }

    def encode_multi_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-representation PromptReps (num_pooled_tokens>0).

        Autoregressively generates tokens, stops at closing quote.
        Mean-pools hidden states (dense), max-pools logits (sparse).

        Returns:
            (dense_emb, sparse_emb)
        """
        device = input_ids.device
        batch_size = input_ids.size(0)

        all_reps = [[] for _ in range(batch_size)]
        all_logits = [[] for _ in range(batch_size)]
        active = [True] * batch_size
        past_key_values = None

        curr_ids = input_ids
        curr_mask = attention_mask

        for step in range(self.num_pooled_tokens):
            active_indices = [i for i, a in enumerate(active) if a]
            if not active_indices:
                break

            outputs = self.model(
                input_ids=curr_ids,
                attention_mask=curr_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            # Next-token logits and hidden states
            next_logits = outputs.logits[:, -1, :]
            next_hidden = outputs.hidden_states[-1][:, -1, :]
            next_token_ids = next_logits.argmax(dim=-1).unsqueeze(-1)

            # Check for stop token (closing quote)
            valid_indices = []
            for idx, active_idx in enumerate(active_indices):
                token_str = self.tokenizer.decode(next_token_ids[idx])
                if '"' in token_str:
                    active[active_idx] = False
                else:
                    all_reps[active_idx].append(next_hidden[idx])
                    all_logits[active_idx].append(next_logits[idx])
                    valid_indices.append(idx)

            if not valid_indices:
                break

            # Filter to still-active sequences for next step.
            # Newer transformers returns DynamicCache; older returns tuple-of-tuples.
            pkv = past_key_values
            if isinstance(pkv, DynamicCache):
                # key_cache/value_cache are read-only properties in newer transformers;
                # use update() to populate the new cache layer-by-layer.
                new_cache = DynamicCache()
                for layer_idx in range(len(pkv.key_cache)):
                    new_cache.update(
                        pkv.key_cache[layer_idx][valid_indices],
                        pkv.value_cache[layer_idx][valid_indices],
                        layer_idx,
                    )
                new_past = new_cache
            else:
                new_past = tuple(
                    tuple(kv[valid_indices] for kv in layer)
                    for layer in pkv
                )
            new_ids = torch.cat([next_token_ids[i] for i in valid_indices], dim=0).unsqueeze(1)
            new_mask = torch.stack([
                torch.cat([curr_mask[i], torch.ones(1, device=device)])
                for i in valid_indices
            ])

            curr_ids = new_ids
            curr_mask = new_mask
            past_key_values = new_past

        # Aggregate: mean-pool hidden states, max-pool logits
        dense_list = []
        sparse_list = []
        for i in range(batch_size):
            if all_reps[i]:
                dense_list.append(torch.stack(all_reps[i]).mean(dim=0))
                stacked_logits = torch.stack(all_logits[i])
                sparse_list.append(stacked_logits.max(dim=0).values)
            else:
                # Fallback: no tokens generated, use zeros
                h_size = outputs.hidden_states[-1].size(-1)
                v_size = outputs.logits.size(-1)
                dense_list.append(torch.zeros(h_size, device=device))
                sparse_list.append(torch.zeros(v_size, device=device))

        dense = torch.stack(dense_list)
        sparse = torch.stack(sparse_list)
        sparse = torch.log(1 + torch.relu(sparse))

        if self.normalize:
            dense = F.normalize(dense, p=2, dim=-1)

        return dense, sparse

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        is_query: bool = True,
        batch_size: int = 8,
        encode_type: str = 'all',
        sparse_topk: int = 128,
        strict: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode texts into dense and/or sparse representations.

        Args:
            encode_type: 'all' (both), 'dense', 'sparse', or 'all_steps'
                         (per-position repr_hidden + quotation_emb + sparse, requires
                         num_pooled_tokens > 0).
            sparse_topk: top-k entries to keep in per-position sparse (all_steps mode).

        Returns:
            Dict with keys depending on encode_type:
              - 'all'/'dense'/'sparse': 'dense' and/or 'sparse'
              - 'all_steps' (num_pooled_tokens>0): 'repr_hidden', 'quotation_emb',
                'sparse_indices', 'sparse_values'
        """
        self.eval()
        device = next(self.model.parameters()).device
        all_dense = []
        all_sparse = []
        # accumulators for all_steps keys
        acc: Dict[str, list] = {}

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = self._tokenize_with_prefix_suffix(batch_texts, is_query)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            if self.num_pooled_tokens > 0 and encode_type == 'all_steps':
                if strict:
                    batch_result = self.encode_multi_token_all_steps(
                        encoded['input_ids'], encoded['attention_mask'],
                        sparse_topk=sparse_topk,
                        texts=batch_texts,
                    )
                else:
                    try:
                        batch_result = self.encode_multi_token_all_steps(
                            encoded['input_ids'], encoded['attention_mask'],
                            sparse_topk=sparse_topk,
                            texts=batch_texts,
                        )
                    except RuntimeError as e:
                        # Flash Attention 2 bug (_prepare_from_posids / cu_seq_lens empty) on
                        # small/last batches — fall back to per-sample encoding.
                        logger.warning(
                            f"Batch of {len(batch_texts)} failed ({e}), retrying per-sample"
                        )
                        per_sample = {}
                        for j, txt in enumerate(batch_texts):
                            try:
                                r = self.encode_multi_token_all_steps(
                                    encoded['input_ids'][j:j+1],
                                    encoded['attention_mask'][j:j+1],
                                    sparse_topk=sparse_topk,
                                    texts=[txt],
                                )
                                for k, v in r.items():
                                    per_sample.setdefault(k, []).append(v.cpu())
                            except RuntimeError as e2:
                                logger.warning(f"  Skipping sample {i+j}: {e2}")
                        if not per_sample:
                            continue
                        batch_result = {k: torch.cat(vs, dim=0) for k, vs in per_sample.items()}
                for k, v in batch_result.items():
                    acc.setdefault(k, []).append(v.cpu())
                continue

            # Content token IDs for sparse filtering (stopwords removed)
            batch_content_ids = None
            if encode_type in ('all', 'sparse', 'all_steps'):
                batch_content_ids = get_content_token_ids(batch_texts, self.tokenizer)

            if self.num_pooled_tokens > 0:
                dense, sparse = self.encode_multi_token(
                    encoded['input_ids'], encoded['attention_mask']
                )
            else:
                # For single-representation mode (K=1), 'all_steps' is equivalent to 'all'
                _et = 'all' if encode_type == 'all_steps' else encode_type
                dense, sparse = self.encode_single_token(
                    encoded['input_ids'], encoded['attention_mask'],
                    encode_type=_et,
                    content_token_ids=batch_content_ids,
                )

            if dense is not None:
                all_dense.append(dense.cpu())
            if sparse is not None:
                all_sparse.append(sparse.cpu())

        # Return all_steps result
        if acc:
            return {k: torch.cat(vs, dim=0) for k, vs in acc.items()}

        result = {}
        if all_dense:
            result['dense'] = torch.cat(all_dense, dim=0)
        if all_sparse:
            result['sparse'] = torch.cat(all_sparse, dim=0)
        return result
