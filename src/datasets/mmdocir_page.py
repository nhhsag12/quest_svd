from __future__ import annotations

import json

import pandas as pd

from .common import RetrievalSample


def build_mmdocir_page_qa_pairs(
    annotations_path: str,
    pages_parquet: str,
    all_page_indices: list[int],
) -> list[RetrievalSample]:
    pages_df = pd.read_parquet(pages_parquet)

    embedded_rows = pages_df.iloc[all_page_indices].copy()
    embedded_rows["embed_idx"] = list(range(len(all_page_indices)))
    embedded_rows["join_doc_name"] = embedded_rows["doc_name"].astype(str).str.replace(".pdf", "", regex=False)
    embedded_rows["passage_id"] = embedded_rows["passage_id"].astype(str)

    doc_page_lookup = {doc: grp for doc, grp in embedded_rows.groupby("join_doc_name")}
    avail_docs = set(doc_page_lookup.keys())

    qa_pairs: list[RetrievalSample] = []

    with open(annotations_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                doc_data = json.loads(line.strip())
            except Exception:
                continue

            doc_name = str(doc_data.get("doc_name", "")).replace(".pdf", "")
            domain = str(doc_data.get("domain", "General"))

            if doc_name not in avail_docs:
                continue

            doc_pages = doc_page_lookup[doc_name]
            doc_all_embed_idxs = sorted(doc_pages["embed_idx"].tolist())
            global_to_local = {g: l for l, g in enumerate(doc_all_embed_idxs)}

            for qa in doc_data.get("questions", []):
                question_text = str(qa.get("Q", "")).strip()
                if not question_text:
                    continue

                gt_page_ids = qa.get("page_id", [])
                if not isinstance(gt_page_ids, list):
                    gt_page_ids = [gt_page_ids]
                gt_page_ids_str = [str(p) for p in gt_page_ids]

                gt_embed_idxs = []
                for pid_str in gt_page_ids_str:
                    matched = doc_pages[doc_pages["passage_id"] == pid_str]
                    gt_embed_idxs.extend(matched["embed_idx"].tolist())

                gt_embed_idxs = sorted(set(gt_embed_idxs))
                if not gt_embed_idxs:
                    continue

                gt_local_idxs = [global_to_local[g] for g in gt_embed_idxs if g in global_to_local]
                if not gt_local_idxs:
                    continue

                qa_pairs.append(
                    RetrievalSample(
                        question=question_text,
                        gt_embed_indices=gt_embed_idxs,
                        gt_local_indices=gt_local_idxs,
                        doc_embed_indices=doc_all_embed_idxs,
                        doc_name=doc_name,
                        domain=domain,
                    )
                )

    return qa_pairs
