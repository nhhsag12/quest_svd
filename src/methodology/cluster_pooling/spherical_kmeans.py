from __future__ import annotations

import torch
import torch.nn.functional as F


def spherical_kmeans(x: torch.Tensor, k: int, n_iters: int = 10):
    n, _ = x.shape
    k = min(k, n)

    if k >= n:
        return x.clone()

    perm = torch.randperm(n, device=x.device)
    centroids = x[perm[:k]].clone()

    for _ in range(n_iters):
        sim = torch.mm(x, centroids.t())
        cluster_ids = sim.argmax(dim=1)

        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            members = (cluster_ids == c).nonzero(as_tuple=True)[0]
            new_centroids[c] = x[members].mean(dim=0) if members.numel() > 0 else centroids[c]

        centroids = F.normalize(new_centroids, dim=-1)

    return centroids
