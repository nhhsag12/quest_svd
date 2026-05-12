from __future__ import annotations

import torch
import torch.nn.functional as F


SVD_RANK_REMOVE = 1
TEMPERATURE_OURS = 0.5


def remove_sink_components_batch(attn_heads: torch.Tensor, rank_remove: int):
    try:
        u_mat, s_vec, vh_mat = torch.linalg.svd(attn_heads, full_matrices=False)
        rank_remove = min(rank_remove, s_vec.shape[-1])
        sink = (u_mat[..., :rank_remove] * s_vec[:, :rank_remove].unsqueeze(1)) @ vh_mat[:, :rank_remove, :]
        return (attn_heads - sink).clamp(min=0.0)
    except Exception:
        return attn_heads


def compute_svd_importance_softplus(attentions, content_mask: torch.Tensor, layer_weights: torch.Tensor, rank_remove: int = SVD_RANK_REMOVE):
    device = content_mask.device
    batch_size, seq_len = content_mask.shape
    importance = torch.zeros(batch_size, seq_len, device=device)

    for i, attn in enumerate(attentions):
        attn = attn.float().to(device)
        b_sz, n_heads, seq_q, seq_k = attn.shape
        attn_flat = attn.view(b_sz * n_heads, seq_q, seq_k)
        cleaned = remove_sink_components_batch(attn_flat, rank_remove)
        cleaned = cleaned.view(b_sz, n_heads, seq_q, seq_k)

        layer_imp = cleaned.sum(dim=2).mean(dim=1)
        layer_imp = layer_imp * content_mask
        layer_imp = layer_imp / layer_imp.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        importance += layer_weights[i] * layer_imp

    importance = importance * content_mask
    importance = F.softplus(importance / TEMPERATURE_OURS)
    importance = importance * content_mask
    return importance


def build_content_mask_qwen(inputs, processor):
    attn_mask = inputs["attention_mask"]
    input_ids = inputs.get("input_ids", None)
    if input_ids is None:
        return attn_mask.float()

    tok = getattr(processor, "tokenizer", processor)
    special_ids = set()
    for attr in ["pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id", "sep_token_id", "cls_token_id"]:
        tid = getattr(tok, attr, None)
        if tid is not None:
            special_ids.add(int(tid))

    if hasattr(tok, "added_tokens_encoder"):
        for _, tid in tok.added_tokens_encoder.items():
            special_ids.add(int(tid))

    if not special_ids:
        return attn_mask.float()

    special_tensor = torch.tensor(list(special_ids), device=input_ids.device)
    is_special = (input_ids.unsqueeze(-1) == special_tensor).any(dim=-1)
    return attn_mask.float() * (~is_special).float()
