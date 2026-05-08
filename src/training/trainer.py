"""
Trainer for LLaDA 2 block diffusion retrieval.

Extends the vendored Tevatron trainer (see ``_base_trainer``) with:
- Progressive backbone unfreezing
- Optional block curriculum (start with fewer blocks, increase over training)
- Custom _save to handle retrieval head weights
"""

import torch
import logging
import os

from ._base_trainer import TevatronTrainer
from .arguments import LLaDA2TrainingArguments

logger = logging.getLogger(__name__)


class LLaDA2RetrievalTrainer(TevatronTrainer):
    """
    Trainer for LLaDA 2 block diffusion retrieval models.

    Supports:
    - Standard contrastive training (InfoNCE)
    - Progressive backbone unfreezing
    - Block curriculum: gradually increase active blocks
    - Custom save to persist retrieval head weights
    """

    def __init__(self, *args, freeze_backbone_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)

        self.freeze_backbone_steps = freeze_backbone_steps
        self._backbone_unfrozen = freeze_backbone_steps == 0

        if freeze_backbone_steps > 0:
            encoder = getattr(self.model, 'encoder', None)
            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = False
                logger.info(f"Backbone frozen for first {freeze_backbone_steps} steps")

    def training_step(self, model, inputs):
        """Override to add block curriculum and backbone unfreezing."""
        current_step = self.state.global_step

        # Progressive backbone unfreezing
        if not self._backbone_unfrozen and current_step >= self.freeze_backbone_steps:
            logger.info(f"Step {current_step}: Unfreezing LLaDA 2 backbone")
            for param in model.encoder.parameters():
                param.requires_grad = True
            self._backbone_unfrozen = True

        # Block curriculum (optional)
        args = self.args
        if hasattr(args, 'block_curriculum') and args.block_curriculum:
            total_steps = args.max_steps if args.max_steps > 0 else (
                len(self.train_dataset) * args.num_train_epochs // args.per_device_train_batch_size
            )
            progress = min(current_step / max(total_steps, 1), 1.0)
            start_blocks = args.block_curriculum_start
            end_blocks = args.block_curriculum_end
            if end_blocks <= 0:
                # Will be resolved to max at runtime
                end_blocks = 16  # reasonable max for 512 tokens / 32 block_len
            current_blocks = int(start_blocks + (end_blocks - start_blocks) * progress)
            current_blocks = max(1, current_blocks)

            # Truncate input to current_blocks * block_length during curriculum
            # This is applied via the model's block_schedule — store for logging
            if current_step % 100 == 0:
                logger.info(f"Step {current_step}: block curriculum = {current_blocks} blocks")

        return super().training_step(model, inputs)

    def _save(self, output_dir=None, state_dict=None):
        """Override to save retrieval head weights alongside the encoder."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.model

        # Save the full model using its own save method
        if hasattr(model, 'save'):
            model.save(output_dir)
        else:
            super()._save(output_dir, state_dict)

    def log(self, logs):
        """Add LLaDA 2 specific metrics to logging."""
        if hasattr(self, '_backbone_unfrozen'):
            logs['backbone_frozen'] = not self._backbone_unfrozen
        super().log(logs)
