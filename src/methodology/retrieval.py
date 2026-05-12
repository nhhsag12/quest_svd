from __future__ import annotations

import torch
import torch.nn.functional as F


def build_doc_matrix(embeddings, device: str):
    arrays = [torch.from_numpy(e).float() for e in embeddings]
    max_len = max(a.shape[0] for a in arrays)
    dim = arrays[0].shape[1]
    n_docs = len(arrays)

    mat = torch.zeros(n_docs, max_len, dim, dtype=torch.float32)
    mask = torch.zeros(n_docs, max_len, dtype=torch.bool)

    for i, arr in enumerate(arrays):
        length = arr.shape[0]
        mat[i, :length] = F.normalize(arr, dim=-1)
        mask[i, :length] = True

    return mat.to(device), mask.to(device)


@torch.no_grad()
def fast_maxsim(q_norm: torch.Tensor, doc_matrix: torch.Tensor, doc_mask: torch.Tensor) -> torch.Tensor:
    sim = torch.einsum("qd,nld->qnl", q_norm, doc_matrix)
    sim.masked_fill_(~doc_mask.unsqueeze(0), float("-inf"))
    return sim.max(dim=-1).values
