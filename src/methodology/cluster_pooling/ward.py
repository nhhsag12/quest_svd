from __future__ import annotations

import torch
import torch.nn.functional as F

from methodology.retrieval import fast_maxsim


def _ward_distance_matrix(cents: torch.Tensor, sizes: torch.Tensor):
    device = cents.device
    sim = torch.matmul(cents, cents.t()).clamp(-1.0, 1.0)
    sq_dist = 2.0 * (1.0 - sim)
    ni = sizes.float().unsqueeze(1)
    nj = sizes.float().unsqueeze(0)
    w = (ni * nj) / (ni + nj)
    ward = w * sq_dist
    mask = torch.ones(cents.shape[0], cents.shape[0], dtype=torch.bool, device=device).tril()
    ward.masked_fill_(mask, float("inf"))
    return ward


def ward_pool(vecs: torch.Tensor, n_clusters: int):
    n_tokens = vecs.shape[0]
    if n_tokens <= n_clusters:
        return vecs.clone()

    device = vecs.device
    sums = vecs.clone().float()
    sizes = torch.ones(n_tokens, device=device, dtype=torch.float32)
    cents = F.normalize(sums, dim=-1)
    active = torch.ones(n_tokens, dtype=torch.bool, device=device)
    cur_clusters = n_tokens

    while cur_clusters > n_clusters:
        active_idx = active.nonzero(as_tuple=True)[0]
        c_act = cents[active_idx]
        s_act = sizes[active_idx]
        ward = _ward_distance_matrix(c_act, s_act)
        flat_idx = ward.argmin().item()
        n_act = active_idx.shape[0]
        ai, aj = flat_idx // n_act, flat_idx % n_act
        gi, gj = active_idx[ai].item(), active_idx[aj].item()

        sums[gi] = sums[gi] + sums[gj]
        sizes[gi] = sizes[gi] + sizes[gj]
        cents[gi] = F.normalize(sums[gi].unsqueeze(0), dim=-1).squeeze(0)
        active[gj] = False
        cur_clusters -= 1

    final_idx = active.nonzero(as_tuple=True)[0]
    return cents[final_idx]


def ward_pool_scores_all_ratios(q_norm: torch.Tensor, doc_matrix: torch.Tensor, doc_mask: torch.Tensor, topk_ratios: list[float]):
    n = q_norm.shape[0]
    results = {}
    token_counts = {}
    prev_vecs, prev_c = q_norm.clone(), n

    for ratio in sorted(topk_ratios, reverse=True):
        target_c = max(1, round(n * ratio))
        if target_c >= prev_c:
            centroids = prev_vecs
        else:
            centroids = ward_pool(prev_vecs, target_c)
            prev_vecs, prev_c = centroids, centroids.shape[0]

        m_c = fast_maxsim(centroids, doc_matrix, doc_mask)
        results[ratio] = m_c.sum(0)
        token_counts[ratio] = centroids.shape[0]

    return results, token_counts
