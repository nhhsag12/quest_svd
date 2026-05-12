from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_SRC = Path(__file__).resolve().parents[1]
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from methodology.config import ExperimentConfig
from methodology.data_loading import build_qa_pairs, load_page_embeddings
from methodology.reporting import print_summary, save_summary_csv
from methodology.retrieval import build_doc_matrix
from model.colsmol import load_query_model_and_processor


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--page-pkl-dir", required=True)
    parser.add_argument("--colsmol-base", required=True)
    parser.add_argument("--colsmol-lora", required=True)
    parser.add_argument("--annotations-path", required=True)
    parser.add_argument("--pages-parquet", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    return parser


def load_runtime(args):
    config = ExperimentConfig(
        page_pkl_dir=Path(args.page_pkl_dir),
        colsmol_base=Path(args.colsmol_base),
        colsmol_lora=Path(args.colsmol_lora),
        annotations_path=Path(args.annotations_path),
        pages_parquet=Path(args.pages_parquet),
        output_dir=Path(args.output_dir),
        device=args.device,
        kmeans_iters=getattr(args, "kmeans_iters", 10),
        n_random_seeds=getattr(args, "n_random_seeds", 1),
    )
    config.ensure_output_dir()

    all_page_embeddings, all_page_indices = load_page_embeddings(str(config.page_pkl_dir))
    qa_pairs = build_qa_pairs(str(config.annotations_path), str(config.pages_parquet), all_page_indices)
    doc_matrix, doc_mask = build_doc_matrix(all_page_embeddings, config.device)
    query_model, query_processor = load_query_model_and_processor(
        str(config.colsmol_base),
        str(config.colsmol_lora),
        config.device,
    )
    return config, qa_pairs, doc_matrix, doc_mask, query_model, query_processor


def load_runtime_with_embeddings(args):
    config = ExperimentConfig(
        page_pkl_dir=Path(args.page_pkl_dir),
        colsmol_base=Path(args.colsmol_base),
        colsmol_lora=Path(args.colsmol_lora),
        annotations_path=Path(args.annotations_path),
        pages_parquet=Path(args.pages_parquet),
        output_dir=Path(args.output_dir),
        device=args.device,
        kmeans_iters=getattr(args, "kmeans_iters", 10),
        n_random_seeds=getattr(args, "n_random_seeds", 1),
    )
    config.ensure_output_dir()

    all_page_embeddings, all_page_indices = load_page_embeddings(str(config.page_pkl_dir))
    qa_pairs = build_qa_pairs(str(config.annotations_path), str(config.pages_parquet), all_page_indices)
    query_model, query_processor = load_query_model_and_processor(
        str(config.colsmol_base),
        str(config.colsmol_lora),
        config.device,
    )
    return config, qa_pairs, all_page_embeddings, all_page_indices, query_model, query_processor


def finalize_report(metrics, domain_metrics, keys, topk_eval, output_dir: Path, title: str, prefix: str):
    print_summary(metrics, keys, topk_eval, title=title)
    save_summary_csv(metrics, domain_metrics, keys, topk_eval, output_dir, prefix)
    print(f"Saved: {prefix}_summary.csv and {prefix}_domain.csv")
