from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass

from datasets.mmdocir_page import build_mmdocir_page_qa_pairs


@dataclass
class QAPair:
    question: str
    gt_embed_indices: list[int]
    gt_local_indices: list[int]
    doc_embed_indices: list[int]
    doc_name: str
    domain: str


def load_page_embeddings(page_pkl_dir: str):
    pkl_files = sorted(glob.glob(os.path.join(page_pkl_dir, "*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in {page_pkl_dir}")

    all_page_embeddings = []
    all_page_indices = []

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as file:
            data = pickle.load(file)
        if isinstance(data, tuple) and len(data) == 2:
            embs, idxs = data
        elif isinstance(data, dict):
            embs = data.get("embeddings", data.get("encoded_quote", []))
            idxs = data.get("indices", data.get("quote_indices", list(range(len(embs)))))
        else:
            raise ValueError(f"Unknown pkl format in {pkl_path}")

        all_page_embeddings.extend(list(embs))
        all_page_indices.extend(list(idxs))

    return all_page_embeddings, all_page_indices


def build_qa_pairs(annotations_path: str, pages_parquet: str, all_page_indices: list[int]) -> list[QAPair]:
    samples = build_mmdocir_page_qa_pairs(annotations_path, pages_parquet, all_page_indices)
    return [
        QAPair(
            question=s.question,
            gt_embed_indices=s.gt_embed_indices,
            gt_local_indices=s.gt_local_indices,
            doc_embed_indices=s.doc_embed_indices,
            doc_name=s.doc_name,
            domain=s.domain,
        )
        for s in samples
    ]
