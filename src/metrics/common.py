from __future__ import annotations

import numpy as np


def top_k_indices(scores, k: int) -> list[int]:
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if k <= len(scores):
        return [index for index, _ in indexed_scores[:k]]
    return [index for index, _ in indexed_scores]


def compute_ndcg(top_k: list[int], gt_set: set[int], k: int) -> float:
    dcg = sum(1.0 / np.log2(rank + 2) for rank, idx in enumerate(top_k[:k]) if idx in gt_set)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def first_hit(top_k: list[int], gt_set: set[int]) -> int:
    for rank, idx in enumerate(top_k):
        if idx in gt_set:
            return rank + 1
    return -1


def hit_metrics(top_indices: list[int], gt_local: list[int], topk_list=(1, 3, 5, 10)) -> dict[str, float]:
    gt_set = set(gt_local)
    h = first_hit(top_indices, gt_set)

    result: dict[str, float] = {}
    for k in topk_list:
        result[f"r{k}"] = float(h != -1 and h <= k)
        result[f"n{k}"] = float(compute_ndcg(top_indices, gt_set, k))
    return result
