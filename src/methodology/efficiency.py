from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch


class EfficiencyTracker:
    def __init__(self, method_name: str, dim: int = 128):
        self.name = method_name
        self.dim = dim
        self.data: dict[float, list[dict[str, float]]] = {}

    def add(self, ratio: float, n_q_tokens: int, n_doc_tokens: int, n_pages: int):
        if ratio not in self.data:
            self.data[ratio] = []
        flops = 2 * n_q_tokens * n_doc_tokens * self.dim
        self.data[ratio].append({"n_q": n_q_tokens, "n_d": n_doc_tokens, "n_pg": n_pages, "flops": flops})

    def to_dataframe(self):
        rows = []
        baseline_flops = None
        if 1.0 in self.data:
            baseline_flops = np.mean([d["flops"] for d in self.data[1.0]])

        for ratio in sorted(self.data.keys()):
            entries = self.data[ratio]
            nqs = [d["n_q"] for d in entries]
            avg_fl = np.mean([d["flops"] for d in entries])
            row = {
                "method": self.name,
                "ratio": ratio,
                "n_queries": len(entries),
                "avg_Nq": round(np.mean(nqs), 2),
                "std_Nq": round(np.std(nqs), 2),
                "avg_Nd": round(np.mean([d["n_d"] for d in entries]), 0),
                "avg_FLOPs": round(avg_fl, 0),
                "avg_GFLOPs": round(avg_fl / 1e9, 6),
            }
            if baseline_flops and baseline_flops > 0:
                row["FLOPs_pct"] = round(avg_fl / baseline_flops * 100, 2)
                row["speedup"] = round(baseline_flops / avg_fl, 4)
            else:
                row["FLOPs_pct"] = None
                row["speedup"] = None
            rows.append(row)
        return pd.DataFrame(rows)


class ThroughputBenchmark:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.query_pool: dict[float, list[torch.Tensor]] = {}

    def collect(self, ratio: float, q_vecs: torch.Tensor):
        if ratio not in self.query_pool:
            self.query_pool[ratio] = []
        self.query_pool[ratio].append(q_vecs.detach().to(self.device))

    @torch.no_grad()
    def run(self, doc_matrix, doc_mask, n_warmup: int = 10, n_reps: int = 50, batch_size: int = 32):
        n_docs = doc_matrix.shape[0]
        dim = doc_matrix.shape[2]
        total_d_tok = doc_mask.sum().item()
        results = []

        for ratio in sorted(self.query_pool.keys()):
            pool = self.query_pool[ratio]
            if not pool:
                continue

            n_q_sizes = [q.shape[0] for q in pool]
            max_nq = max(n_q_sizes)
            avg_nq = np.mean(n_q_sizes)

            padded = []
            q_masks = []
            for q in pool:
                nq = q.shape[0]
                if nq < max_nq:
                    padded.append(torch.cat([q, torch.zeros(max_nq - nq, dim, device=self.device)]))
                else:
                    padded.append(q[:max_nq])
                m = torch.zeros(max_nq, device=self.device, dtype=torch.bool)
                m[: min(nq, max_nq)] = True
                q_masks.append(m)

            q_bank = torch.stack(padded)
            mask_bank = torch.stack(q_masks)
            pool_size = q_bank.shape[0]
            bs = min(batch_size, pool_size)

            def _score_batch(idxs):
                for i in idxs:
                    q_valid = q_bank[i][mask_bank[i]]
                    if q_valid.shape[0] == 0:
                        continue
                    sim = torch.einsum("qd,nld->qnl", q_valid, doc_matrix)
                    sim.masked_fill_(~doc_mask.unsqueeze(0), float("-inf"))
                    sim.max(dim=-1).values.sum(dim=0)

            for _ in range(n_warmup):
                _score_batch(torch.randint(0, pool_size, (bs,)))
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_q = 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_reps):
                idxs = torch.randint(0, pool_size, (bs,))
                _score_batch(idxs)
                total_q += bs
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            qps = total_q / elapsed
            ms_per_q = (elapsed / total_q) * 1000.0
            flops_pq = 2 * avg_nq * total_d_tok * dim
            gflops_s = (flops_pq * qps) / 1e9

            results.append(
                {
                    "ratio": ratio,
                    "avg_Nq": round(avg_nq, 1),
                    "n_docs": n_docs,
                    "batch_size": bs,
                    "n_reps": n_reps,
                    "total_queries": total_q,
                    "elapsed_s": round(elapsed, 3),
                    "queries_per_sec": round(qps, 1),
                    "ms_per_query": round(ms_per_q, 3),
                    "GFLOPs_per_sec": round(gflops_s, 2),
                }
            )

        return pd.DataFrame(results)
