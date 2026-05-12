from __future__ import annotations

import torch
from peft import PeftModel
from torch import nn
from transformers.models.qwen2_vl import Qwen2VLModel, Qwen2VLProcessor


class ColQwen2(Qwen2VLModel):
    def __init__(self, config, projection_dim: int = 128):
        super().__init__(config)
        self.dim = projection_dim
        self.linear = nn.Linear(self.config.hidden_size, self.dim)

    def forward(self, *args, **kwargs):
        outputs = Qwen2VLModel.forward(self, *args, **kwargs)
        last_hidden_states = outputs[0]
        proj = self.linear(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj

    def forward_with_attentions(self, *args, **kwargs):
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        raw_outputs = Qwen2VLModel.forward(
            self,
            *args,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )
        last_hidden_states = raw_outputs[0]
        proj = self.linear(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj, raw_outputs


class ColQwen2Processor(Qwen2VLProcessor):
    """Processor for ColQwen2."""


def load_query_model_and_processor(colqwen2_base: str, colqwen2_lora: str, device: str):
    query_model = ColQwen2.from_pretrained(
        colqwen2_base,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    try:
        query_model = PeftModel.from_pretrained(query_model, colqwen2_lora).eval()
    except Exception:
        query_model = query_model.eval()

    query_processor = ColQwen2Processor.from_pretrained(colqwen2_lora)
    return query_model, query_processor
