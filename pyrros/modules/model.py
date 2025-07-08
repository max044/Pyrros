'''Utilities to load a causal language model with optional 4‑bit quantisation and QLoRA fine‑tuning adapters.'''

from __future__ import annotations

import logging
from typing import Tuple, Union, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # peft is optional
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

logger = logging.getLogger(__name__)

__all__: list[str] = [
    'load_model',
]


def _maybe_get_peft_model(model, r: int, alpha: int, dropout: float):
    '''Wrap the base model with a PEFT‑LoRA adapter if PEFT is available.'''
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError('PEFT is not installed – run `pip install peft` to enable QLoRA/LoRA support.')

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    return get_peft_model(model, lora_cfg)


@torch.inference_mode()
def load_model(
    name_or_path: str,
    *,
    use_qlora: bool = False,
    bnb4bit: bool = False,
    device_map: Union[str, Dict[str, int]] = 'auto',
    gradient_checkpointing: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    tokenizer_kwargs: Dict[str, Any] | None = None,
    model_kwargs: Dict[str, Any] | None = None,
) -> Tuple['torch.nn.Module', 'transformers.PreTrainedTokenizer']:
    '''Load a HF CausalLM & tokenizer with optional 4‑bit quantisation and QLoRA.'''

    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}

    # 1) Tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True, **tokenizer_kwargs)
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Quantisation config ------------------------------------------------
    quant_cfg = None
    if bnb4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.info('Loading model in 4‑bit nf4 quantisation mode.')

    # 3) Base model ---------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if not bnb4bit else None,
        quantization_config=quant_cfg,
        **model_kwargs,
    )

    # 4) Optional QLoRA -----------------------------------------------------
    if use_qlora:
        logger.info('Enabling QLoRA adapters (r=%d, alpha=%d, dropout=%.2f).', lora_r, lora_alpha, lora_dropout)
        model = _maybe_get_peft_model(model, lora_r, lora_alpha, lora_dropout)

    # 5) Memory tweaks ------------------------------------------------------
    if gradient_checkpointing or use_qlora:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # necessary when using ckpt
        logger.info('Gradient checkpointing enabled.')

    logger.info("Model '%s' loaded successfully (QLoRA=%s, 4‑bit=%s).", name_or_path, use_qlora, bnb4bit)
    return model, tokenizer
