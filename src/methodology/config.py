from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    page_pkl_dir: Path
    colsmol_base: Path
    colsmol_lora: Path
    annotations_path: Path
    pages_parquet: Path
    output_dir: Path
    topk_ratios: list[float] = field(default_factory=lambda: [round(i * 0.1, 1) for i in range(1, 10)])
    n_last_layers_list: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    normalize_modes: list[str] = field(default_factory=lambda: ["pre", "post"])
    attn_n_layers_list: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    kmeans_iters: int = 10
    n_random_seeds: int = 1
    topk_eval: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    device: str = "cuda"

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class RuntimePaths:
    all_page_embeddings: list
    all_page_indices: list[int]
