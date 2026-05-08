"""
Backbone adapters for diffusion retriever training.

Each adapter encapsulates ALL model-specific behavior in one place:
  - How to load the backbone (AutoModel vs AutoModelForCausalLM)
  - PEFT/LoRA configuration (target modules, task type)
  - Attention mask format (2D vs 4D)
  - Hidden state extraction (forward hook on output projection)
  - Mask token ID (verified from HuggingFace tokenizer configs)
  - Gradient checkpointing support

The TrainableDiffusionRetriever delegates to an adapter and has ZERO
model-specific branches.  Adding a new model = adding one adapter class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class BackboneAdapter(ABC):
    """Abstract interface for model-specific backbone behavior."""

    model_type: str          # e.g. 'dream', 'llada1', 'llada2'
    mask_token_id: int       # verified from HuggingFace model cards
    hub_model_name: str      # HuggingFace model ID for fallback loading

    def __init__(self):
        self.flash_attn: bool = False   # set by load_backbone

    # -- loading --------------------------------------------------------

    def load_backbone(self, source: str, device_map=None) -> nn.Module:
        """Load backbone from HuggingFace model name or local directory.

        Tries flash attention variants in order, falls back to eager.
        Sets ``self.flash_attn`` as a side effect.
        """
        common_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
        if device_map is not None:
            common_kw['device_map'] = device_map

        for attn_impl in self._flash_attn_impls:
            try:
                bb = self._auto_class().from_pretrained(
                    source, attn_implementation=attn_impl, **common_kw)
                self.flash_attn = True
                logger.info(f"{self.model_type}: {attn_impl} enabled")
                return bb
            except (ValueError, ImportError):
                pass

        self.flash_attn = False
        return self._auto_class().from_pretrained(source, **common_kw)

    @staticmethod
    @abstractmethod
    def _auto_class():
        """Return the AutoModel class to use (AutoModel or AutoModelForCausalLM)."""

    _flash_attn_impls: Tuple[str, ...] = ('flash_attention_2',)

    # -- PEFT / LoRA ----------------------------------------------------

    @abstractmethod
    def get_lora_config(self, lora_rank: int, lora_alpha: int,
                        lora_dropout: float = 0.0):
        """Return a ``peft.LoraConfig`` appropriate for this backbone."""

    # -- attention mask --------------------------------------------------

    @abstractmethod
    def needs_4d_mask(self) -> bool:
        """Whether the backbone expects a 4D ``[B,1,L,L]`` attention mask.

        If False, the backbone handles bidirectional attention internally
        and expects a standard 2D ``[B,L]`` padding mask.
        """

    # -- hidden state extraction -----------------------------------------

    def register_hidden_hook(self, backbone: nn.Module,
                             ref_dict: Dict[str, torch.Tensor]) -> bool:
        """Register a forward hook on the output projection to capture the
        last hidden state without ``output_hidden_states=True``.

        Returns True if a hook was registered, False otherwise (in which
        case the caller should fall back to ``output_hidden_states``).
        """
        return False  # default: no hook, use output_hidden_states

    # -- gradient checkpointing ------------------------------------------

    def enable_gradient_checkpointing(self, backbone: nn.Module, **kwargs):
        """Enable gradient checkpointing.  Override for models that don't
        support it."""
        backbone.gradient_checkpointing_enable(**kwargs)
        logger.info("Gradient checkpointing enabled")


# ---------------------------------------------------------------------------
# Hook helpers (shared across adapters)
# ---------------------------------------------------------------------------

def _is_linear(mod: nn.Module) -> bool:
    """Check if a module is a Linear layer (plain or LoRA-wrapped)."""
    if isinstance(mod, nn.Linear):
        return True
    # PEFT LoRA wraps nn.Linear in peft.tuners.lora.layer.Linear which is
    # NOT a subclass of nn.Linear, but has a base_layer that is.
    if hasattr(mod, 'base_layer') and isinstance(mod.base_layer, nn.Linear):
        return True
    return False


def _hook_on_module(backbone: nn.Module, ref_dict: Dict,
                    target_name: str, skip_if_contains: Optional[str] = None,
                    adapter_name: str = '') -> bool:
    """Register a forward hook on the first Linear (or LoRA-wrapped Linear)
    whose leaf name matches *target_name*.  Optionally skip modules whose
    full path contains *skip_if_contains* (e.g. 'blocks' to skip per-layer
    ff_out).
    """
    for name, mod in backbone.named_modules():
        leaf = name.split('.')[-1]
        if leaf == target_name and _is_linear(mod):
            if skip_if_contains and skip_if_contains in name:
                continue
            mod.register_forward_hook(
                lambda m, inp, out, r=ref_dict: r.update({'h': inp[0]})
            )
            logger.info(f"{adapter_name}: hook on '{name}'")
            return True
    return False


# ---------------------------------------------------------------------------
# Dream
# ---------------------------------------------------------------------------

class DreamAdapter(BackboneAdapter):
    model_type = 'dream'
    mask_token_id = 151666          # <|mask|>  — Dream-org/Dream-v0-Instruct-7B
    hub_model_name = 'Dream-org/Dream-v0-Instruct-7B'

    @staticmethod
    def _auto_class():
        from transformers import AutoModel
        return AutoModel

    def get_lora_config(self, lora_rank, lora_alpha, lora_dropout=0.0):
        from peft import LoraConfig, TaskType
        # Dream is loaded via AutoModel (not AutoModelForCausalLM), so
        # FEATURE_EXTRACTION avoids PeftModelForCausalLM which would
        # require prepare_inputs_for_generation (Dream lacks this).
        return LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none",
        )

    def needs_4d_mask(self) -> bool:
        # Standard HF Qwen2 attention — needs 4D to enforce bidirectional.
        return True

    def register_hidden_hook(self, backbone, ref_dict):
        return _hook_on_module(backbone, ref_dict, 'lm_head',
                               adapter_name='dream')


# ---------------------------------------------------------------------------
# LLaDA v1 (GSAI-ML/LLaDA-8B-Instruct)
# ---------------------------------------------------------------------------

class LLaDA1Adapter(BackboneAdapter):
    model_type = 'llada1'
    mask_token_id = 126336          # <|mdm_mask|>  — GSAI-ML/LLaDA-8B-Instruct
    hub_model_name = 'GSAI-ML/LLaDA-8B-Instruct'
    _flash_attn_impls = ('flash_attention_3', 'flash_attention_2')

    @staticmethod
    def _auto_class():
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM

    def get_lora_config(self, lora_rank, lora_alpha, lora_dropout=0.0):
        from peft import LoraConfig, TaskType
        # LLaDA1 custom arch uses different names than standard LLaMA:
        #   attn_out (not o_proj), ff_proj (not gate_proj), ff_out (not down_proj)
        return LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out",
                            "ff_proj", "up_proj", "ff_out"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    def needs_4d_mask(self) -> bool:
        # Custom code handles bidirectional attention internally — always 2D.
        return False

    def register_hidden_hook(self, backbone, ref_dict):
        # LLaDA1 has ff_out per-block (blocks.X.ff_out = FFN output) AND at
        # the model level (transformer.ff_out = output projection to vocab).
        # We need the model-level one; skip per-block ones.
        return _hook_on_module(backbone, ref_dict, 'ff_out',
                               skip_if_contains='blocks',
                               adapter_name='llada1')

    def enable_gradient_checkpointing(self, backbone, **kwargs):
        # LLaDA1's LLaDAModelLM doesn't support HF gradient_checkpointing_enable.
        # Manually wrap each transformer block with torch checkpoint.
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        # Find the blocks ModuleList through the PEFT wrapper
        blocks = None
        for name, mod in backbone.named_modules():
            if name.endswith('.blocks') and isinstance(mod, nn.ModuleList):
                blocks = mod
                break

        if blocks is None:
            logger.warning("LLaDA1: couldn't find transformer blocks — "
                           "skipping gradient checkpointing")
            return

        for block in blocks:
            orig_forward = block.forward

            def _make_ckpt(fwd):
                def _ckpt_forward(*args, **kwargs):
                    if not torch.is_grad_enabled():
                        return fwd(*args, **kwargs)
                    return ckpt_fn(fwd, *args, use_reentrant=False, **kwargs)
                return _ckpt_forward

            block.forward = _make_ckpt(orig_forward)

        logger.info(f"LLaDA1: manual gradient checkpointing on {len(blocks)} blocks")


# ---------------------------------------------------------------------------
# LLaDA v1.5 (GSAI-ML/LLaDA-1.5)  — same architecture as v1
# ---------------------------------------------------------------------------

class LLaDA15Adapter(LLaDA1Adapter):
    model_type = 'llada15'
    mask_token_id = 126336          # <|mdm_mask|>  — same tokenizer as v1
    hub_model_name = 'GSAI-ML/LLaDA-1.5'

    def register_hidden_hook(self, backbone, ref_dict):
        return _hook_on_module(backbone, ref_dict, 'ff_out',
                               skip_if_contains='blocks',
                               adapter_name='llada15')



# ---------------------------------------------------------------------------
# LLaDA v2 (inclusionAI/LLaDA2.0-mini)
# ---------------------------------------------------------------------------

class LLaDA2Adapter(BackboneAdapter):
    model_type = 'llada2'
    mask_token_id = 156895          # <|mask|>  — inclusionAI/LLaDA2.0-mini
    hub_model_name = 'inclusionAI/LLaDA2.0-mini'
    _flash_attn_impls = ('flash_attention_3', 'flash_attention_2')

    @staticmethod
    def _auto_class():
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM

    def get_lora_config(self, lora_rank, lora_alpha, lora_dropout=0.0):
        from peft import LoraConfig, TaskType
        # LLaDA2 uses fused QKV ("query_key_value") and "dense" for attn
        # output.  Skip MoE expert FFN layers to avoid multiplying params.
        return LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=[
                "query_key_value", "dense",
                "mlp.shared_experts.gate_proj",
                "mlp.shared_experts.up_proj",
                "mlp.shared_experts.down_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    def needs_4d_mask(self) -> bool:
        # Standard HF causal model — needs 4D to override causal attention,
        # unless flash attention handles masking itself.
        return not self.flash_attn

    def register_hidden_hook(self, backbone, ref_dict):
        return _hook_on_module(backbone, ref_dict, 'lm_head',
                               adapter_name='llada2')


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: Dict[str, type] = {
    'dream':  DreamAdapter,
    'llada1': LLaDA1Adapter,
    'llada15': LLaDA15Adapter,
    'llada2': LLaDA2Adapter,
}


def get_adapter(model_type: str) -> BackboneAdapter:
    """Create a BackboneAdapter for the given model_type."""
    cls = ADAPTER_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Available: {sorted(ADAPTER_REGISTRY.keys())}")
    return cls()
