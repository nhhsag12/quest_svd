from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalSample:
    question: str
    gt_embed_indices: list[int]
    gt_local_indices: list[int]
    doc_embed_indices: list[int]
    doc_name: str
    domain: str
    gt_relevance: dict[int, float] | None = None
    layout_mapping_raw: list[dict] | None = None
    candidate_bbox_list: list[tuple[int, float, float, float, float] | None] | None = None
