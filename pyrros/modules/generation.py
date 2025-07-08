"""Response generation utilities (HF + optional vLLM)."""
from __future__ import annotations

from typing import List, Sequence, Dict, Any, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = ["generate_responses"]


@torch.inference_mode()
def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_vllm: bool = False,
    vllm_engine: Any | None = None,
    generation_kwargs: Dict[str, Any] | None = None,
) -> List[str]:
    """Generate model responses for a batch of *prompts*.

    If *use_vllm* is True, *vllm_engine* should be an instance of ``vllm.LLM``.
    Otherwise, we fall back to standard ``model.generate``.
    """

    generation_kwargs = generation_kwargs or {}

    if use_vllm:
        if vllm_engine is None:
            raise ValueError("`use_vllm=True` but `vllm_engine` is None.")

        outputs = vllm_engine.generate(
            prompts,
            sampling_params=dict(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )
        responses = [out.outputs[0].text for out in outputs]
        return responses

    # HF path ---------------------------------------------------------------
    inputs = tokenizer(list(prompts), return_tensors="pt", padding=True).to(model.device)
    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )
    responses = tokenizer.batch_decode(gen_out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return responses
