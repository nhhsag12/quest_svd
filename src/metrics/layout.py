from __future__ import annotations

from metrics.common import compute_ndcg, first_hit


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
    top = min(b, d)
    left = min(a, c)
    bottom = max(b, d)
    right = max(a, c)
    return top, left, bottom, right


def calculate_overlap_area(a, b) -> float:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    it = max(at, bt)
    il = max(al, bl)
    ib = min(ab, bb)
    ir = min(ar, br)
    if ib <= it or ir <= il:
        return 0.0
    return (ib - it) * (ir - il)


def recall_layout_area(
    top_k_positions: list[int],
    candidate_bbox_list: list[tuple[int, float, float, float, float] | None],
    layout_mapping_raw: list[dict],
) -> float:
    if not layout_mapping_raw or not candidate_bbox_list:
        return 0.0

    recall_area = 0.0
    for pos in top_k_positions:
        if pos < 0 or pos >= len(candidate_bbox_list):
            continue
        candidate = candidate_bbox_list[pos]
        if candidate is None:
            continue

        page_id, t, l, b, r = candidate
        cand_bbox = (t, l, b, r)

        for gt in layout_mapping_raw:
            if not isinstance(gt, dict):
                continue
            gt_page = gt.get("page", gt.get("page_id", None))
            try:
                gt_page = int(float(gt_page))
            except Exception:
                gt_page = None
            if gt_page is not None and gt_page != page_id:
                continue

            gt_bbox_raw = _parse_bbox(gt.get("bbox"))
            if gt_bbox_raw is None:
                continue
            recall_area += calculate_overlap_area(cand_bbox, _to_tlbr(gt_bbox_raw))

    gt_area = 0.0
    for gt in layout_mapping_raw:
        if not isinstance(gt, dict):
            continue
        gt_bbox_raw = _parse_bbox(gt.get("bbox"))
        if gt_bbox_raw is None:
            continue
        t, l, b, r = _to_tlbr(gt_bbox_raw)
        gt_area += max(0.0, b - t) * max(0.0, r - l)

    if gt_area <= 0.0:
        return 0.0
    return min(recall_area / gt_area, 1.0)


def hit_metrics_layout(
    top_indices: list[int],
    gt_local: list[int],
    topk_list=(1, 5, 10),
    candidate_bbox_list: list[tuple[int, float, float, float, float] | None] | None = None,
    layout_mapping_raw: list[dict] | None = None,
) -> dict[str, float]:
    gt_set = set(gt_local)
    h = first_hit(top_indices, gt_set)
    result: dict[str, float] = {}

    for k in topk_list:
        result[f"r{k}"] = float(h != -1 and h <= k)
        result[f"n{k}"] = float(compute_ndcg(top_indices, gt_set, k))

    if candidate_bbox_list is None:
        candidate_bbox_list = []
    if layout_mapping_raw is None:
        layout_mapping_raw = []

    for k in topk_list:
        result[f"recall{k}"] = recall_layout_area(
            top_indices[:k],
            candidate_bbox_list,
            layout_mapping_raw,
        )

    return result
