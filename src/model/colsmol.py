from __future__ import annotations

import time
from types import SimpleNamespace

import torch
from peft import PeftModel
from torch import nn
from transformers import Idefics3Model, Idefics3PreTrainedModel
from colpali_engine.models import ColIdefics3Processor


class ColIdefics3(Idefics3PreTrainedModel):
    def __init__(self, config, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.model: Idefics3Model = Idefics3Model(config)
        self.dim = 128
        self.linear = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.main_input_name = "doc_input_ids"
        self.post_init()

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        last_hidden_states = outputs[0]
        proj = self.linear(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj

    def forward_with_attentions(self, *args, **kwargs):
        kwargs.pop("output_attentions", None)
        raw_outputs = self.model(*args, output_attentions=True, **kwargs)
        last_hidden_states = raw_outputs[0]
        proj = self.linear(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj, raw_outputs


def load_query_model_and_processor(colsmol_base: str, colsmol_lora: str, device: str):
    query_model = ColIdefics3.from_pretrained(
        colsmol_base,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    query_model = PeftModel.from_pretrained(query_model, colsmol_lora).eval()
    query_processor = ColIdefics3Processor.from_pretrained(colsmol_lora)
    return query_model, query_processor


_cached_text_layers_page = None


def _find_text_layers_page(model, force_rescan: bool = False):
    global _cached_text_layers_page
    if _cached_text_layers_page is not None and not force_rescan:
        return _cached_text_layers_page

    candidates = []
    for name, mod in model.named_modules():
        if not name.endswith(".layers"):
            continue
        if not hasattr(mod, "__len__") or len(mod) == 0:
            continue
        is_vision = "vision" in name.lower() or "encoder" in name.lower()
        has_mlp = hasattr(mod[0], "mlp") or hasattr(mod[0], "feed_forward")
        candidates.append({"name": name, "module": mod, "n_layers": len(mod), "is_vision": is_vision, "has_mlp": has_mlp})

    text_c = [c for c in candidates if not c["is_vision"] and c["has_mlp"]]
    best = max(text_c or candidates, key=lambda c: c["n_layers"], default=None)

    if best is None:
        raise RuntimeError("Cannot find transformer text layers.")

    _cached_text_layers_page = best["module"]
    return _cached_text_layers_page


class AttentionCollectorPage:
    def __init__(self):
        self.attentions = []
        self.hooks = []

    @staticmethod
    def _find_attn_module(layer):
        for attr in ["self_attn", "attention", "attn"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

    def register_hooks(self, model, layer_indices: list[int]):
        self.clear()
        self.remove_hooks()

        layers = _find_text_layers_page(model)
        n_total = len(layers)

        for idx in layer_indices:
            idx_abs = idx if idx >= 0 else n_total + idx
            if not (0 <= idx_abs < n_total):
                continue
            attn_mod = self._find_attn_module(layers[idx_abs])
            if attn_mod is None:
                continue

            def _hook(module, inp, out):
                if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                    self.attentions.append(out[1].detach())

            self.hooks.append(attn_mod.register_forward_hook(_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear(self):
        self.attentions.clear()


def _unwrap_to_colidefics3(model):
    if isinstance(model, ColIdefics3):
        return model

    seen = {id(model)}
    queue = [model]
    for _ in range(6):
        if not queue:
            break
        nxt_queue = []
        for cur in queue:
            for attr in ("base_model", "model"):
                if not hasattr(cur, attr):
                    continue
                child = getattr(cur, attr)
                if isinstance(child, ColIdefics3):
                    return child
                cid = id(child)
                if cid not in seen and isinstance(child, nn.Module):
                    seen.add(cid)
                    nxt_queue.append(child)
        queue = nxt_queue
    return None


def forward_with_attentions_colpali(model, inputs):
    kwargs = dict(inputs)
    base = _unwrap_to_colidefics3(model)
    if base is not None and hasattr(base, "forward_with_attentions"):
        with torch.no_grad():
            proj, raw_outputs = base.forward_with_attentions(**kwargs)
        return proj, raw_outputs

    collector = AttentionCollectorPage()
    layers = _find_text_layers_page(model)
    collector.register_hooks(model, list(range(len(layers))))

    try:
        with torch.no_grad():
            kwargs["output_attentions"] = True
            proj = model(**kwargs)
    finally:
        collector.remove_hooks()

    attns = tuple(collector.attentions)
    raw_outputs = SimpleNamespace(attentions=attns if attns else None)
    collector.clear()
    return proj, raw_outputs


def encode_query_live(question: str, processor, model, device: str):
    q_text = f"Query: {question}" + "<pad>" * 10
    inputs = processor(text=[q_text], return_tensors="pt", padding="longest").to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        proj, raw_outputs = forward_with_attentions_colpali(model, inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    encode_ms = (time.perf_counter() - t0) * 1000.0
    return proj, raw_outputs, inputs, encode_ms
