"""
Utilities to load a causal language model with optional 4-bit quantisation,
QLoRA fine-tuning adapters **et** un wrapper Composer prêt à l’emploi.

• Fonctionne en production (poids pré-entraînés, CUDA, bfloat16, QLoRA…)
• Fonctionne en CI / smoke-test (poids aléatoires, CPU/MPS, fp32)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import transformers
from pyrros.models.grpo_model import GRPOModel

# — PEFT (facultatif) --------------------------------------------------------
try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # LoRA/QLoRA optionnels
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


logger = logging.getLogger(__name__)

__all__: list[str] = ["load_model"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _maybe_get_peft_model(model, r: int, alpha: int, dropout: float):
    """Wrap `model` with PEFT-LoRA adapters si PEFT est dispo."""
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError(
            "PEFT n’est pas installé – `pip install peft` pour activer QLoRA/LoRA."
        )

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora_cfg)


# --------------------------------------------------------------------------- #
# API principale                                                              #
# --------------------------------------------------------------------------- #
def load_model(
    name_or_path: str,
    *,
    # --- Fonctionnement général ---------------------------------------------
    pretrained: bool = True,  # False → init aléatoire : idéal pour tests/CI
    device: Union[str, torch.device, None] = None,  # "cuda:0" / "cpu" / "mps"
    dtype: Union[torch.dtype, None] = None,  # fp32 / bf16 / etc.
    device_map: Union[str, Dict[str, int]] = "auto",  # ignoré si `device` set
    # --- Quantisation / LoRA -------------------------------------------------
    bnb4bit: bool = False,
    use_qlora: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # --- Misc hyper-params ---------------------------------------------------
    gradient_checkpointing: bool = False,
    tokenizer_kwargs: Dict[str, Any] | None = None,
    model_kwargs: Dict[str, Any] | None = None,
) -> Tuple["GRPOModel", "transformers.PreTrainedTokenizer"]:
    """
    Charge un modèle CausalLM (HF) + son tokenizer, avec options :

    • `pretrained=False`  → configuration + poids aléatoires (smoke-test quick)
    • `bnb4bit=True`      → BitsAndBytes 4-bit nf4
    • `use_qlora=True`    → ajout d’adapters LoRA
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}

    # 1) ──────────────────── Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True, **tokenizer_kwargs)
    tokenizer.padding_side, tokenizer.truncation_side = "right", "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) ──────────────────── Config quantisation
    quant_cfg = None
    if bnb4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.info("Chargement en 4-bit nf4 (BitsAndBytes) activé.")

    # 3) ──────────────────── Base model
    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            device_map=device_map if device is None else None,
            torch_dtype=dtype
            or (torch.bfloat16 if (not bnb4bit and str(device_map).startswith("cuda")) else torch.float32),
            quantization_config=quant_cfg,
            **model_kwargs,
        )
    else:
        cfg = AutoConfig.from_pretrained(name_or_path)
        model = AutoModelForCausalLM.from_config(cfg, **model_kwargs)

    # Placement explicite si demandé
    if device is not None:
        model.to(device)

    # 4) ──────────────────── (Q)LoRA
    if use_qlora:
        logger.info("Ajout d’adapters QLoRA (r=%d, α=%d, dropout=%.2f).", lora_r, lora_alpha, lora_dropout)
        model = _maybe_get_peft_model(model, lora_r, lora_alpha, lora_dropout)

    # 5) ──────────────────── Gradient checkpointing
    if gradient_checkpointing or use_qlora:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing activé.")

    # 6) ──────────────────── Wrapper GRPO
    model = GRPOModel(model=model, tokenizer=tokenizer)
        

    logger.info(
        "Modèle « %s » prêt (pretrained=%s, QLoRA=%s, 4-bit=%s, Composer=%s).",
        name_or_path,
        pretrained,
        use_qlora,
        bnb4bit,
    )
    return model, tokenizer
