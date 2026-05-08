"""
Arguments for LLaDA 2 block diffusion retrieval.

Builds on the vendored Tevatron base argument classes (see ``_base_arguments``).
"""

from dataclasses import dataclass, field
from typing import Optional

from ._base_arguments import ModelArguments, TevatronTrainingArguments


@dataclass
class LLaDA2ModelArguments(ModelArguments):
    """Model arguments for LLaDA 2 block diffusion retrieval."""

    model_name_or_path: str = field(
        default="inclusionAI/LLaDA2.0-mini",
        metadata={"help": "HuggingFace model name or path. Must be a LLaDA 2 model."}
    )

    # Embedding configuration (no projection — embedding dim = hidden_size)
    pooling: str = field(
        default="mean",
        metadata={"help": "Pooling strategy: mean, weighted_mean, last, attention"}
    )
    normalize: bool = field(
        default=True,
        metadata={"help": "L2-normalize output embeddings"}
    )

    # Block diffusion configuration
    encoding_mode: str = field(
        default="clean",
        metadata={"help": "Encoding mode: clean, block_interactive, promptreps, block_denoising"}
    )
    block_length: int = field(
        default=32,
        metadata={"help": "Number of tokens per block (LLaDA 2 default: 32)"}
    )
    num_steps_per_block: int = field(
        default=8,
        metadata={"help": "Inner denoising steps within each block"}
    )
    enable_t2t: bool = field(
        default=False,
        metadata={"help": "Enable token-to-token editing (LLaDA 2.1 feature)"}
    )
    block_aggregation: str = field(
        default="ema",
        metadata={"help": "Block embedding aggregation: last, mean, weighted_mean, ema, attention"}
    )

    # LLaDA 2 mask token
    mask_token_id: int = field(
        default=156895,
        metadata={"help": "LLaDA 2's <|mask|> token id"}
    )

    # PromptReps configuration
    num_repr_tokens: int = field(
        default=1,
        metadata={"help": "Number of [MASK] representation tokens for promptreps mode"}
    )
    num_denoise_steps: int = field(
        default=1,
        metadata={"help": "Denoising steps for promptreps (1=single pass, >1=iterative refinement)"}
    )
    query_prefix: str = field(
        default="",
        metadata={"help": "Prompt prefix for queries (string or path to file)"}
    )
    query_suffix: str = field(
        default="",
        metadata={"help": "Prompt suffix for queries (string or path to file)"}
    )
    passage_prefix: str = field(
        default="",
        metadata={"help": "Prompt prefix for passages (string or path to file)"}
    )
    passage_suffix: str = field(
        default="",
        metadata={"help": "Prompt suffix for passages (string or path to file)"}
    )

    # Training strategy
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Freeze LLaDA 2 backbone parameters"}
    )
    freeze_backbone_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps with backbone frozen (0=no warmup freeze)"}
    )


@dataclass
class LLaDA2TrainingArguments(TevatronTrainingArguments):
    """Training arguments for LLaDA 2 block diffusion retrieval."""

    temperature: float = field(
        default=0.02,
        metadata={"help": "Temperature for InfoNCE contrastive loss"}
    )

    # Block curriculum: start training with fewer blocks, increase over training
    block_curriculum: bool = field(
        default=False,
        metadata={"help": "Gradually increase number of active blocks during training"}
    )
    block_curriculum_start: int = field(
        default=1,
        metadata={"help": "Starting number of active blocks for curriculum"}
    )
    block_curriculum_end: int = field(
        default=0,
        metadata={"help": "Final number of active blocks (0 = use all blocks)"}
    )
