from __future__ import annotations


def _init_metric(topk_eval, extra_fields=None):
    metric = {"count": 0}
    for k in topk_eval:
        metric[f"r{k}"] = 0.0
        metric[f"n{k}"] = 0.0
    if extra_fields:
        for name in extra_fields:
            metric[name] = 0.0
    return metric


def _ensure_metric(store, key, topk_eval, extra_fields=None):
    if key not in store:
        store[key] = _init_metric(topk_eval, extra_fields=extra_fields)
    return store[key]


def _add_metric(dst, src):
    for field, value in src.items():
        if field == "count":
            continue
        if field not in dst:
            dst[field] = 0.0
        dst[field] += float(value)
    dst["count"] += 1


def record_metric(all_metrics, all_domain_metrics, key, metric, domain, topk_eval):
    extra_fields = [f for f in metric.keys() if f not in {"count"}]
    _add_metric(_ensure_metric(all_metrics, key, topk_eval, extra_fields=extra_fields), metric)
    if domain not in all_domain_metrics:
        all_domain_metrics[domain] = {}
    _add_metric(_ensure_metric(all_domain_metrics[domain], key, topk_eval, extra_fields=extra_fields), metric)
