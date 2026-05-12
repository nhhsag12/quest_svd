from __future__ import annotations

from pathlib import Path

import pandas as pd


def print_summary(all_metrics, method_keys, topk_eval, title: str = ""):
    if title:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    header = f"{'Method':<35}"
    for k in topk_eval:
        header += f" {'R@' + str(k):>7}"
    for k in topk_eval:
        header += f" {'nDCG@' + str(k):>9}"

    extra_fields = []
    for key in method_keys:
        metric = all_metrics.get(key)
        if not metric:
            continue
        for field in metric.keys():
            if field in {"count"}:
                continue
            if field.startswith("r") and field[1:].isdigit():
                continue
            if field.startswith("n") and field[1:].isdigit():
                continue
            if field not in extra_fields:
                extra_fields.append(field)
    for field in extra_fields:
        header += f" {field:>10}"

    print(header)
    print("-" * len(header))

    for key in method_keys:
        if key not in all_metrics:
            continue
        metric = all_metrics[key]
        cnt = metric["count"] or 1
        row = f"{key:<35}"
        for k in topk_eval:
            row += f" {metric[f'r{k}'] / cnt * 100:6.2f}%"
        for k in topk_eval:
            row += f" {metric[f'n{k}'] / cnt:8.4f}"
        for field in extra_fields:
            row += f" {metric.get(field, 0.0) / cnt:9.4f}"
        print(row)


def save_summary_csv(metrics, domain_metrics, method_keys, topk_eval, output_dir: Path, prefix: str):
    rows = []
    for key in method_keys:
        if key not in metrics:
            continue
        metric = metrics[key]
        cnt = metric["count"] or 1
        row = {"method": key, "count": cnt}
        for k in topk_eval:
            row[f"r@{k}"] = round(metric[f"r{k}"] / cnt * 100, 4)
            row[f"ndcg@{k}"] = round(metric[f"n{k}"] / cnt, 6)
        for field, value in metric.items():
            if field in {"count"}:
                continue
            if field.startswith("r") and field[1:].isdigit():
                continue
            if field.startswith("n") and field[1:].isdigit():
                continue
            row[field] = round(float(value) / cnt, 6)
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)

    domain_rows = []
    for domain in sorted(domain_metrics):
        dm = domain_metrics[domain]
        row = {"domain": domain}
        for key in method_keys:
            if key not in dm:
                continue
            metric = dm[key]
            cnt = metric["count"] or 1
            for k in topk_eval:
                row[f"{key}_r{k}"] = round(metric[f"r{k}"] / cnt * 100, 4)
                row[f"{key}_ndcg{k}"] = round(metric[f"n{k}"] / cnt, 6)
            for field, value in metric.items():
                if field in {"count"}:
                    continue
                if field.startswith("r") and field[1:].isdigit():
                    continue
                if field.startswith("n") and field[1:].isdigit():
                    continue
                row[f"{key}_{field}"] = round(float(value) / cnt, 6)
        domain_rows.append(row)

    pd.DataFrame(domain_rows).to_csv(output_dir / f"{prefix}_domain.csv", index=False)
