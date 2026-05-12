from __future__ import annotations

import json

import pandas as pd

from .common import RetrievalSample


def _parse_bbox(raw) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            return float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
        except Exception:
            return None
    if isinstance(raw, dict):
        keys = [
            ("x1", "y1", "x2", "y2"),
            ("top", "left", "bottom", "right"),
            ("left", "top", "right", "bottom"),
        ]
        for k1, k2, k3, k4 in keys:
            if all(k in raw for k in (k1, k2, k3, k4)):
                try:
                    return float(raw[k1]), float(raw[k2]), float(raw[k3]), float(raw[k4])
                except Exception:
                    return None
    return None


def _to_tlbr(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    a, b, c, d = bbox
    # Allow either x1,y1,x2,y2 or top,left,bottom,right and convert to top,left,bottom,right
    top = min(b, d)
    left = min(a, c)
    bottom = max(b, d)
    right = max(a, c)
    return top, left, bottom, right


def _iou_tlbr(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    it = max(at, bt)
    il = max(al, bl)
    ib = min(ab, bb)
    ir = min(ar, br)
    if ib <= it or ir <= il:
        return 0.0
    inter = (ib - it) * (ir - il)
    area_a = max(0.0, ab - at) * max(0.0, ar - al)
    area_b = max(0.0, bb - bt) * max(0.0, br - bl)
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def build_mmdocir_layout_qa_pairs(
    annotations_path: str,
    layouts_parquet: str,
    all_layout_indices: list[int],
    iou_threshold: float = 0.5,
) -> list[RetrievalSample]:
    layouts_df = pd.read_parquet(layouts_parquet)

    embedded_rows = layouts_df.iloc[all_layout_indices].copy()
    embedded_rows["embed_idx"] = list(range(len(all_layout_indices)))

    if "join_doc_name" in embedded_rows.columns:
        embedded_rows["join_doc_name"] = embedded_rows["join_doc_name"].astype(str)
    else:
        embedded_rows["join_doc_name"] = embedded_rows["doc_name"].astype(str).str.replace(".pdf", "", regex=False)

    if "layout_id" not in embedded_rows.columns and "layout" in embedded_rows.columns:
        embedded_rows["layout_id"] = embedded_rows["layout"]

    page_col = "page_id" if "page_id" in embedded_rows.columns else "passage_id"
    embedded_rows[page_col] = embedded_rows[page_col].astype(str)

    doc_layout_lookup = {doc: grp for doc, grp in embedded_rows.groupby("join_doc_name")}
    avail_docs = set(doc_layout_lookup.keys())

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

            doc_layouts = doc_layout_lookup[doc_name]
            doc_all_embed_idxs = sorted(doc_layouts["embed_idx"].tolist())
            global_to_local = {g: l for l, g in enumerate(doc_all_embed_idxs)}

            candidate_bbox_by_global: dict[int, tuple[int, float, float, float, float] | None] = {}
            for _, row in doc_layouts.iterrows():
                g_idx = int(row["embed_idx"])
                page_val = str(row.get(page_col, ""))
                try:
                    page_int = int(float(page_val))
                except Exception:
                    page_int = -1

                bbox = _parse_bbox(row.get("bbox"))
                if bbox is None:
                    bbox = _parse_bbox(row.get("layout_bbox"))

                if bbox is None:
                    candidate_bbox_by_global[g_idx] = None
                    continue

                t, l, b, r = _to_tlbr(bbox)
                candidate_bbox_by_global[g_idx] = (page_int, t, l, b, r)

            for qa in doc_data.get("questions", []):
                question_text = str(qa.get("Q", "")).strip()
                if not question_text:
                    continue

                layout_mapping_raw = qa.get("layout_mapping") or qa.get("layout") or []
                if not isinstance(layout_mapping_raw, list):
                    layout_mapping_raw = []

                gt_embed_idxs: set[int] = set()

                # Direct layout_id supervision if available.
                gt_layout_ids = qa.get("layout_id", [])
                if not isinstance(gt_layout_ids, list):
                    gt_layout_ids = [gt_layout_ids]
                if gt_layout_ids and "layout_id" in doc_layouts.columns:
                    match = doc_layouts[doc_layouts["layout_id"].astype(str).isin([str(x) for x in gt_layout_ids])]
                    gt_embed_idxs.update(match["embed_idx"].tolist())

                # Area/IoU supervision via layout_mapping entries.
                for gt in layout_mapping_raw:
                    if not isinstance(gt, dict):
                        continue
                    gt_page = gt.get("page", gt.get("page_id", None))
                    try:
                        gt_page = int(float(gt_page))
                    except Exception:
                        gt_page = None

                    gt_bbox = _parse_bbox(gt.get("bbox"))
                    if gt_bbox is None:
                        continue
                    gt_tlbr = _to_tlbr(gt_bbox)

                    for g_idx in doc_all_embed_idxs:
                        candidate = candidate_bbox_by_global.get(g_idx)
                        if candidate is None:
                            continue
                        c_page, t, l, b, r = candidate
                        if gt_page is not None and c_page != gt_page:
                            continue
                        if _iou_tlbr((t, l, b, r), gt_tlbr) >= iou_threshold:
                            gt_embed_idxs.add(g_idx)

                if not gt_embed_idxs:
                    continue

                gt_local_idxs = [global_to_local[g] for g in sorted(gt_embed_idxs) if g in global_to_local]
                if not gt_local_idxs:
                    continue

                candidate_bbox_list = [candidate_bbox_by_global.get(g) for g in doc_all_embed_idxs]

                qa_pairs.append(
                    RetrievalSample(
                        question=question_text,
                        gt_embed_indices=sorted(gt_embed_idxs),
                        gt_local_indices=gt_local_idxs,
                        doc_embed_indices=doc_all_embed_idxs,
                        doc_name=doc_name,
                        domain=domain,
                        layout_mapping_raw=layout_mapping_raw,
                        candidate_bbox_list=candidate_bbox_list,
                    )
                )

    return qa_pairs
