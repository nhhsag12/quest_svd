from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from methodology.config import ExperimentConfig
from methodology.data_loading import load_page_embeddings
from methodology.datasets.mmdocir_page import build_mmdocir_page_qa_pairs
from methodology.methods import (
    run_attention_pruning,
    run_hierarchical,
    run_ours_svd_cluster_pool,
    run_random_pruning,
    run_spherical_kmeans,
    run_traditional,
)
from methodology.reporting import print_summary, save_summary_csv
from methodology.retrieval import build_doc_matrix
from model.colqwen import load_query_model_and_processor as load_colqwen_query_model_and_processor
from model.colsmol import load_query_model_and_processor


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMDocIR page-level retrieval experiments.")
    parser.add_argument("--page-pkl-dir", required=True)
    parser.add_argument("--colsmol-base", required=True)
    parser.add_argument("--colsmol-lora", required=True)
    parser.add_argument("--annotations-path", required=True)
    parser.add_argument("--pages-parquet", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-family", choices=["colsmol", "colqwen"], default="colsmol")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["traditional", "ours", "hierarchical", "attention", "kmeans", "random"],
        choices=["traditional", "ours", "hierarchical", "attention", "kmeans", "random"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--n-random-seeds", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        page_pkl_dir=Path(args.page_pkl_dir),
        colsmol_base=Path(args.colsmol_base),
        colsmol_lora=Path(args.colsmol_lora),
        annotations_path=Path(args.annotations_path),
        pages_parquet=Path(args.pages_parquet),
        output_dir=Path(args.output_dir),
        device=args.device,
        kmeans_iters=args.kmeans_iters,
        n_random_seeds=args.n_random_seeds,
    )
    config.ensure_output_dir()

    if args.dry_run:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}, indent=2))
        return

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no GPU is available.")

    print("Loading page embeddings...")
    all_page_embeddings, all_page_indices = load_page_embeddings(str(config.page_pkl_dir))
    print(f"Loaded {len(all_page_embeddings)} page embeddings")

    print("Building QA pairs...")
    qa_pairs = build_mmdocir_page_qa_pairs(str(config.annotations_path), str(config.pages_parquet), all_page_indices)
    print(f"Total QA pairs: {len(qa_pairs)}")

    print("Building document matrix...")
    doc_matrix, doc_mask = build_doc_matrix(all_page_embeddings, config.device)

    print("Loading query model and processor...")
    if args.model_family == "colqwen":
        query_model, query_processor = load_colqwen_query_model_and_processor(
            str(config.colsmol_base),
            str(config.colsmol_lora),
            config.device,
        )
    else:
        query_model, query_processor = load_query_model_and_processor(
            str(config.colsmol_base),
            str(config.colsmol_lora),
            config.device,
        )

    summary_registry = {}

    if "traditional" in args.methods:
        m, dm = run_traditional(
            qa_pairs,
            query_processor,
            query_model,
            doc_matrix,
            doc_mask,
            config.topk_eval,
            config.device,
            metric_protocol="set",
        )
        keys = ["traditional"]
        print_summary(m, keys, config.topk_eval, title="Traditional MaxSim")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "traditional")
        summary_registry["traditional"] = m

    if "ours" in args.methods:
        m, dm = run_ours_svd_cluster_pool(
            qa_pairs,
            query_processor,
            query_model,
            doc_matrix,
            doc_mask,
            config.topk_eval,
            config.topk_ratios,
            config.n_last_layers_list,
            config.normalize_modes,
            config.device,
            metric_protocol="set",
        )
        keys = sorted(m.keys())
        print_summary(m, keys[:12], config.topk_eval, title="Ours SVD + ClusterPool (first 12 keys)")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "ours_ablation")
        summary_registry["ours"] = m

    if "hierarchical" in args.methods:
        m, dm = run_hierarchical(
            qa_pairs,
            query_processor,
            query_model,
            doc_matrix,
            doc_mask,
            config.topk_eval,
            config.topk_ratios,
            config.device,
            metric_protocol="set",
        )
        keys = ["traditional"] + [f"hier_r{int(r * 100)}" for r in config.topk_ratios]
        print_summary(m, keys, config.topk_eval, title="Hierarchical Ward")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "hierarchical")
        summary_registry["hierarchical"] = m

    if "attention" in args.methods:
        m, dm = run_attention_pruning(
            qa_pairs,
            query_processor,
            query_model,
            doc_matrix,
            doc_mask,
            config.topk_eval,
            config.topk_ratios,
            config.attn_n_layers_list,
            config.device,
            metric_protocol="set",
        )
        keys = ["traditional"] + [f"attn_L1_r{int(r * 100)}_trad" for r in config.topk_ratios]
        print_summary(m, keys, config.topk_eval, title="Attention Pruning (L1 trad)")
        save_summary_csv(m, dm, sorted(m.keys()), config.topk_eval, config.output_dir, "attention_pruning")
        summary_registry["attention"] = m

    if "kmeans" in args.methods:
        m, dm = run_spherical_kmeans(
            qa_pairs,
            query_processor,
            query_model,
            doc_matrix,
            doc_mask,
            config.topk_eval,
            config.topk_ratios,
            config.kmeans_iters,
            config.device,
            metric_protocol="set",
        )
        keys = ["traditional"] + [f"kmeans_r{int(r * 100)}" for r in config.topk_ratios]
        print_summary(m, keys, config.topk_eval, title="Spherical KMeans")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "spherical_kmeans")
        summary_registry["kmeans"] = m

    if "random" in args.methods:
        m, dm = run_random_pruning(
            qa_pairs,
            query_processor,
            query_model,
            doc_matrix,
            doc_mask,
            config.topk_eval,
            config.topk_ratios,
            config.n_random_seeds,
            config.device,
            metric_protocol="set",
        )
        keys = ["traditional"] + [f"rand_r{int(r * 100)}" for r in config.topk_ratios]
        print_summary(m, keys, config.topk_eval, title="Random Pruning")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "random_pruning")
        summary_registry["random"] = m

    print("Done. Artifacts saved to:", config.output_dir)


if __name__ == "__main__":
    main()
