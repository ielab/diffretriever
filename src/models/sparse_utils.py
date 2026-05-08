"""
Sparse representation utilities matching original PromptReps.

Reference: https://github.com/ielab/PromptReps
The original filters sparse logits by:
1. Word-tokenizing the text (lowercased) with NLTK
2. Removing stopwords and punctuation
3. Re-tokenizing each content word independently to get clean token IDs
4. Only keeping logits for those token IDs
"""

import torch
from typing import List, Set, Optional
import string
import logging

logger = logging.getLogger(__name__)

try:
    from nltk import word_tokenize
    from nltk.corpus import stopwords as _sw_corpus
    STOPWORDS = set(_sw_corpus.words('english') + list(string.punctuation))
except LookupError:
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk import word_tokenize
    from nltk.corpus import stopwords as _sw_corpus
    STOPWORDS = set(_sw_corpus.words('english') + list(string.punctuation))


def get_content_token_ids(texts: List[str], tokenizer) -> List[Set[int]]:
    """Extract content token IDs from texts (stopwords removed, word-level tokenization).

    Matches original PromptReps get_valid_tokens_values:
    1. word_tokenize(text.lower())
    2. Remove stopwords + punctuation
    3. tokenizer.encode(word) for each remaining word

    Args:
        texts: List of raw text strings.
        tokenizer: HuggingFace tokenizer.

    Returns:
        List of sets of token IDs, one per text.
    """
    all_token_ids = []
    for text in texts:
        words = [w for w in word_tokenize(text.lower())
                 if w not in STOPWORDS]
        if words:
            # Batch encode all words at once (much faster than per-word encode)
            batch_ids = tokenizer(words, add_special_tokens=False)['input_ids']
            token_ids = set()
            for ids in batch_ids:
                token_ids.update(ids)
        else:
            token_ids = set()
        all_token_ids.append(token_ids)
    return all_token_ids


def filter_sparse(
    sparse: torch.Tensor,
    content_token_ids: List[Set[int]],
    exclude_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """Filter sparse logits to only keep content tokens.

    Args:
        sparse: [batch, vocab] sparse logit tensor.
        content_token_ids: List of sets of valid token IDs per example.
        exclude_ids: Token IDs to always exclude (e.g., MASK token).

    Returns:
        Filtered sparse tensor (same shape, non-content entries zeroed).
    """
    # Build (row, col) index pairs for all content tokens across batch
    rows, cols = [], []
    for i in range(sparse.size(0)):
        if content_token_ids[i]:
            ids = list(content_token_ids[i])
            rows.extend([i] * len(ids))
            cols.extend(ids)
    if rows:
        mask = torch.zeros_like(sparse)
        mask[rows, cols] = 1.0
    else:
        mask = torch.zeros_like(sparse)
    if exclude_ids:
        for eid in exclude_ids:
            mask[:, eid] = 0.0
    return sparse * mask
