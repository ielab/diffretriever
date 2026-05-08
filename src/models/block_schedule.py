"""
Block Schedule for LLaDA 2 Block Diffusion Retrieval

LLaDA 2 uses block diffusion — semi-autoregressive denoising in blocks of
fixed length (default 32 tokens). Within each block, tokens are denoised via
confidence-based selection over multiple steps. Across blocks, generation is
autoregressive (block k depends on blocks 0..k-1).

This replaces the per-token noise schedules from LLaDA 1.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BlockSchedule:
    """Configuration for LLaDA 2's block diffusion process.

    Attributes:
        block_length: Number of tokens per block (default 32 for LLaDA 2).
        num_steps_per_block: Inner denoising steps within each block.
        enable_t2t: Enable token-to-token editing (LLaDA 2.1 feature).
            When True, already-placed tokens can be revised during denoising.
    """

    block_length: int = 32
    num_steps_per_block: int = 8
    enable_t2t: bool = False

    @property
    def num_transfer_tokens(self) -> int:
        """Tokens revealed per inner denoising step within a block."""
        return max(1, self.block_length // self.num_steps_per_block)

    def get_num_blocks(self, seq_len: int) -> int:
        """Return the number of blocks for a given sequence length.

        Rounds up so the last block may be partial.
        """
        return (seq_len + self.block_length - 1) // self.block_length

    def get_block_boundaries(self, seq_len: int) -> List[Tuple[int, int]]:
        """Return (start, end) index pairs for each block.

        The last block is clamped to seq_len.
        """
        boundaries = []
        for i in range(0, seq_len, self.block_length):
            boundaries.append((i, min(i + self.block_length, seq_len)))
        return boundaries
