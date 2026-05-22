"""
Bottleneck Diffusion Retriever — Semantic Hub Training.

This model implements the "Semantic Hub" contribution:
1. Aggressive masking of text tokens (30-50%).
2. Bottleneck Attention: Text tokens are "blind" to each other and must 
   reconstruct themselves by attending to the K retrieval [MASK] tokens.
3. Forces K tokens to become the optimal global semantic summary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
from .diffretriever_trainable import TrainableDiffusionRetriever
from .sparse_utils import filter_sparse

logger = logging.getLogger(__name__)

class BottleneckDiffusionRetriever(TrainableDiffusionRetriever):
    """
    Diffusion Retriever with Semantic Hub Bottlenecking.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottleneck_weight = 0.0
        self.bottleneck_mask_ratio = 0.35

    def _build_bottleneck_mask(self, seq_len: int, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Builds the Semantic Hub Bottleneck Mask:
        - Text tokens CANNOT see other text tokens.
        - Text tokens CAN see themselves and the K Summary tokens.
        - Summary tokens (K masks) CAN see everything (full bidirectional).
        """
        K = self.n_gen_tokens
        mask_4d = self._build_4d_mask(seq_len, attention_mask)
        B = attention_mask.size(0)
        device = attention_mask.device
        dtype = mask_4d.dtype
        min_val = torch.finfo(dtype).min
        
        pos = torch.arange(seq_len, device=device)

        # With left-padding, MASK tokens are always at the end: positions [L-K, L)
        gen_start = seq_len - K - self._n_eos

        # Identify text tokens (those before the K masks, excluding padding)
        is_real = attention_mask.bool()  # [B, L]
        is_text_row = is_real & (pos.unsqueeze(0) < gen_start)  # [B, L]
        is_text_col = is_text_row  # [B, L]
        is_not_self = (pos.unsqueeze(0) != pos.unsqueeze(1)) # [L, L]
        
        # Pattern: If a row is a text token and a col is another text token, block it.
        # This forces the text token to look at the K masks (which are NOT blocked).
        bottleneck_pattern = is_text_row.unsqueeze(2) & is_text_col.unsqueeze(1) & is_not_self.unsqueeze(0)
        mask_4d = mask_4d.masked_fill(bottleneck_pattern.unsqueeze(1), min_val)
        
        return mask_4d

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        passage_input_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
        query_content_ids: Optional[List] = None,
        passage_content_ids: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        # Handle standard retrieval loss via parent
        # We only override to inject the bottleneck auxiliary pass
        
        # Note: We temporarily set denoising_weight to 0 so parent doesn't run standard denoising
        orig_dn_w = self.denoising_weight
        self.denoising_weight = 0.0
        
        # Standard retrieval pass (using parent's forward logic)
        loss_dict = super().forward(
            query_input_ids, query_attention_mask,
            passage_input_ids, passage_attention_mask,
            query_content_ids, passage_content_ids
        )
        
        self.denoising_weight = orig_dn_w

        # --- Semantic Hub Bottleneck Auxiliary Pass ---
        if self.bottleneck_weight > 0:
            # 1. Apply aggressive masking to passages
            # We reuse the masking helper but with our ratio
            orig_ratio = self.denoise_mask_ratio
            self.denoise_mask_ratio = self.bottleneck_mask_ratio
            p_corrupted, p_targets, ratio = self._apply_text_masking(
                passage_input_ids, passage_attention_mask)
            self.denoise_mask_ratio = orig_ratio

            # 2. Build the bottleneck mask
            bn_mask = self._build_bottleneck_mask(passage_input_ids.size(1), passage_attention_mask)

            # 3. Forward pass with the bottleneck constraint
            # This forces hidden states at masked positions to be reconstructed from K tokens
            _, p_logits_bn = self._fwd(p_corrupted, passage_attention_mask, 
                                      need_logits=True, mask_4d=bn_mask)
            
            # 4. Compute reconstruction loss
            bn_loss = self.compute_denoising_loss(p_logits_bn, p_targets, ratio)
            
            # 5. Combine losses
            loss_dict['loss'] = loss_dict['loss'] + self.bottleneck_weight * bn_loss
            loss_dict['loss_bottleneck'] = bn_loss.detach()

        return loss_dict

    def _save_retriever_config(self, output_dir: str):
        super()._save_retriever_config(output_dir)
        # Append bottleneck specific configs
        import json, os
        cfg_path = os.path.join(output_dir, 'retriever_config.json')
        with open(cfg_path, 'r') as f:
            config = json.load(f)
        config['bottleneck_weight'] = self.bottleneck_weight
        config['bottleneck_mask_ratio'] = self.bottleneck_mask_ratio
        config['is_bottleneck_model'] = True
        with open(cfg_path, 'w') as f:
            json.dump(config, f, indent=2)
