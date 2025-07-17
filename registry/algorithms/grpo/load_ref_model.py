from composer.core import Algorithm, Event, State
from pyrros.utils.model_utils import load_model
from composer.distributed import parallelize_composer_model, prepare_tp_module
import torch

from registry.models.grpo.grpo_model import GRPOModel

class LoadRefModel(Algorithm):
    def __init__(self, ref_model_name: str):
        super().__init__(match_event=Event.AFTER_LOAD)
        self.ref_model_name = ref_model_name

    def match(self, event, state):
        return event == Event.AFTER_LOAD

    def apply(self, event, state: State, logger):

        ref_model, tokenizer = load_model(
            self.ref_model_name,
            pretrained=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        ref_model = GRPOModel(model=ref_model, tokenizer=tokenizer)
        ref_model.eval()
        ref_model.requires_grad_(False)
        ref_model.to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        if state.tp_config is not None:
            prepare_tp_module(
                ref_model,
                None,
                state.tp_config,
            )

        if state.fsdp_config is not None and not state.load_monolith_rank0_only:
            parallelize_composer_model(
                ref_model,
                None,
                state.fsdp_config,  # type: ignore
            )

        state.ref_model = ref_model
