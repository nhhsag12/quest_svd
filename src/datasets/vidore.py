from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .common import RetrievalSample


@dataclass(frozen=True)
class VidoreDomainSpec:
    name: str
    queries_files: list[Path]
    qrels_files: list[Path]
    corpus_files: list[Path]


def _norm_key(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def _norm_domain(value) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text.startswith("vidore_v3_"):
        return text.removeprefix("vidore_v3_")
    if text.startswith("vidore_v3-"):
        return text.removeprefix("vidore_v3-")
    return text


def _collect_parquets(base: Path, subdir: str) -> list[Path]:
    return sorted(base.glob(f"**/{subdir}/*.parquet"))


def discover_vidore_domains(vidore_root: str) -> list[VidoreDomainSpec]:
    root = Path(vidore_root)
    if not root.exists():
        raise FileNotFoundError(f"ViDoRe dataset root not found: {vidore_root}")

    domains: list[VidoreDomainSpec] = []
    for outer in sorted(root.glob("vidore_v3_*")):
        if not outer.is_dir():
            continue

        domain_name = outer.name.removeprefix("vidore_v3_")
        candidate_bases = [
            outer / outer.name,
            outer,
        ]

        queries_files: list[Path] = []
        qrels_files: list[Path] = []
        corpus_files: list[Path] = []

        for base in candidate_bases:
            if not queries_files:
                queries_files = sorted(base.glob("queries/*.parquet"))
            if not qrels_files:
                qrels_files = sorted(base.glob("qrels/*.parquet"))
            if not corpus_files:
                corpus_files = sorted(base.glob("corpus/*.parquet"))

        if not queries_files or not qrels_files or not corpus_files:
            continue

        domains.append(
            VidoreDomainSpec(
                name=domain_name,
                queries_files=queries_files,
                qrels_files=qrels_files,
                corpus_files=corpus_files,
            )
        )

    return domains


def _normalize_encoded_rows(encoded_metadata) -> pd.DataFrame:
    if isinstance(encoded_metadata, pd.DataFrame):
        df = encoded_metadata.copy()
    else:
        df = pd.DataFrame(list(encoded_metadata))

    if df.empty:
        raise ValueError("Encoded ViDoRe metadata is empty.")

    if "embed_idx" not in df.columns:
        df = df.copy()
        df["embed_idx"] = range(len(df))

    if "corpus_id" not in df.columns:
        raise ValueError("Encoded ViDoRe metadata must contain a corpus_id column.")

    if "domain" not in df.columns:
        df = df.copy()
        df["domain"] = "unknown"

    df["embed_idx"] = df["embed_idx"].astype(int)
    df["corpus_id_norm"] = df["corpus_id"].map(_norm_key)
    df["domain_norm"] = df["domain"].map(_norm_domain)
    return df


def build_vidore_qa_pairs(
    vidore_root: str,
    encoded_metadata,
    domain_filter: list[str] | None = None,
    dedup_queries: bool = True,
) -> list[RetrievalSample]:
    encoded_df = _normalize_encoded_rows(encoded_metadata)
    domain_filter_norm = {d.lower() for d in domain_filter} if domain_filter else None

    corpus_to_embed_global: dict[str, list[int]] = {}
    corpus_to_embed_domain: dict[tuple[str, str], list[int]] = {}

    for _, row in encoded_df.iterrows():
        corpus_key = row["corpus_id_norm"]
        if not corpus_key:
            continue
        embed_idx = int(row["embed_idx"])
        domain_key = row["domain_norm"]

        corpus_to_embed_global.setdefault(corpus_key, []).append(embed_idx)
        corpus_to_embed_domain.setdefault((domain_key, corpus_key), []).append(embed_idx)

    for key in list(corpus_to_embed_global.keys()):
        corpus_to_embed_global[key] = sorted(set(corpus_to_embed_global[key]))
    for key in list(corpus_to_embed_domain.keys()):
        corpus_to_embed_domain[key] = sorted(set(corpus_to_embed_domain[key]))

    qa_pairs: list[RetrievalSample] = []
    for spec in discover_vidore_domains(vidore_root):
        if domain_filter_norm and spec.name.lower() not in domain_filter_norm:
            continue

        queries_df = pd.concat([pd.read_parquet(path) for path in spec.queries_files], ignore_index=True)
        qrels_df = pd.concat([pd.read_parquet(path) for path in spec.qrels_files], ignore_index=True)

        if queries_df.empty or qrels_df.empty:
            continue

        if "score" in qrels_df.columns:
            qrels_df = qrels_df[pd.to_numeric(qrels_df["score"], errors="coerce") > 0].copy()
        else:
            qrels_df = qrels_df.copy()

        query_id_to_relevance: dict[str, dict[int, float]] = {}
        query_id_to_gt: dict[str, list[int]] = {}

        grouped_qrels = qrels_df.groupby(qrels_df["query_id"].map(_norm_key))
        for qid_key, group in grouped_qrels:
            if not qid_key:
                continue

            gt_embed_idxs: set[int] = set()
            gt_relevance: dict[int, float] = {}

            for _, qrel_row in group.iterrows():
                corpus_key = _norm_key(qrel_row.get("corpus_id"))
                if not corpus_key:
                    continue

                rel_raw = qrel_row.get("score", 1)
                try:
                    rel_score = float(rel_raw)
                except Exception:
                    rel_score = 1.0

                domain_key = spec.name.lower()
                domain_match = corpus_to_embed_domain.get((domain_key, corpus_key), [])
                global_match = corpus_to_embed_global.get(corpus_key, [])
                mapped = domain_match or global_match

                for embed_idx in mapped:
                    gt_embed_idxs.add(int(embed_idx))
                    gt_relevance[int(embed_idx)] = max(gt_relevance.get(int(embed_idx), 0.0), rel_score)

            if gt_embed_idxs:
                query_id_to_gt[qid_key] = sorted(gt_embed_idxs)
                query_id_to_relevance[qid_key] = gt_relevance

        domain_doc_indices = encoded_df.loc[encoded_df["domain_norm"] == spec.name.lower(), "embed_idx"].tolist()
        if not domain_doc_indices:
            domain_doc_indices = encoded_df["embed_idx"].tolist()

        domain_doc_indices = sorted(int(idx) for idx in domain_doc_indices)
        doc_local = {g: local for local, g in enumerate(domain_doc_indices)}

        for _, query_row in queries_df.iterrows():
            qid_key = _norm_key(query_row.get("query_id"))
            if not qid_key:
                continue

            question = str(query_row.get("query", "")).strip()
            if not question:
                continue

            gt_embed_idxs = query_id_to_gt.get(qid_key, [])
            if not gt_embed_idxs:
                continue

            gt_local_idxs = [doc_local[g] for g in gt_embed_idxs if g in doc_local]
            if not gt_local_idxs:
                continue

            qa_pairs.append(
                RetrievalSample(
                    question=question,
                    gt_embed_indices=gt_embed_idxs,
                    gt_local_indices=gt_local_idxs,
                    doc_embed_indices=domain_doc_indices,
                    doc_name=f"vidore::{spec.name}::{qid_key}",
                    domain=spec.name,
                    gt_relevance=query_id_to_relevance.get(qid_key, {}),
                )
            )

    if dedup_queries:
        seen = set()
        deduped: list[RetrievalSample] = []
        for sample in qa_pairs:
            key = (sample.domain, sample.question)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(sample)
        qa_pairs = deduped

    return qa_pairs
