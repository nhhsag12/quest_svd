from __future__ import annotations

import argparse
from pathlib import Path

from methodology.config import ExperimentConfig
from methodology.datasets.mmdocir_layout import build_mmdocir_layout_qa_pairs
from methodology.data_loading import load_page_embeddings
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
from model.colsmol import load_query_model_and_processor as load_colsmol_query_model_and_processor


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMDocIR layout-level retrieval experiments with area recall.")
    parser.add_argument("--layout-pkl-dir", required=True)
    parser.add_argument("--colsmol-base", required=True)
    parser.add_argument("--colsmol-lora", required=True)
    parser.add_argument("--annotations-path", required=True)
    parser.add_argument("--layouts-parquet", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["traditional", "ours", "hierarchical", "attention", "kmeans", "random"],
        choices=["traditional", "ours", "hierarchical", "attention", "kmeans", "random"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-family", choices=["colsmol", "colqwen"], default="colsmol")
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--n-random-seeds", type=int, default=1)
    parser.add_argument("--layout-iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        page_pkl_dir=Path(args.layout_pkl_dir),
        colsmol_base=Path(args.colsmol_base),
        colsmol_lora=Path(args.colsmol_lora),
        annotations_path=Path(args.annotations_path),
        pages_parquet=Path(args.layouts_parquet),
        output_dir=Path(args.output_dir),
        device=args.device,
        kmeans_iters=args.kmeans_iters,
        n_random_seeds=args.n_random_seeds,
    )
    config.ensure_output_dir()

    all_layout_embeddings, all_layout_indices = load_page_embeddings(str(config.page_pkl_dir))
    qa_pairs = build_mmdocir_layout_qa_pairs(
        str(config.annotations_path),
        str(config.pages_parquet),
        all_layout_indices,
        iou_threshold=args.layout_iou_threshold,
    )
    print(f"Total layout QA pairs: {len(qa_pairs)}")

    doc_matrix, doc_mask = build_doc_matrix(all_layout_embeddings, config.device)
    if args.model_family == "colqwen":
        query_model, query_processor = load_colqwen_query_model_and_processor(
            str(config.colsmol_base),
            str(config.colsmol_lora),
            config.device,
        )
    else:
        query_model, query_processor = load_colsmol_query_model_and_processor(
            str(config.colsmol_base),
            str(config.colsmol_lora),
            config.device,
        )

    metric_protocol = "layout_area"

    if "traditional" in args.methods:
        m, dm = run_traditional(
            qa_pairs,
            query_processor,
            query_model,
            doc_matrix,
            doc_mask,
            config.topk_eval,
            config.device,
            metric_protocol=metric_protocol,
        )
        print_summary(m, ["traditional"], config.topk_eval, title="MMDocIR Layout - Traditional")
        save_summary_csv(m, dm, ["traditional"], config.topk_eval, config.output_dir, "mmdocir_layout_traditional")

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
            metric_protocol=metric_protocol,
        )
        keys = sorted(m.keys())
        print_summary(m, keys[:12], config.topk_eval, title="MMDocIR Layout - Ours (first 12 keys)")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "mmdocir_layout_ours")

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
            metric_protocol=metric_protocol,
        )
        keys = ["traditional"] + [f"hier_r{int(r * 100)}" for r in config.topk_ratios]
        print_summary(m, keys, config.topk_eval, title="MMDocIR Layout - Hierarchical")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "mmdocir_layout_hierarchical")

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
            metric_protocol=metric_protocol,
        )
        keys = sorted(m.keys())
        print_summary(m, keys[:12], config.topk_eval, title="MMDocIR Layout - Attention (first 12 keys)")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "mmdocir_layout_attention")

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
            metric_protocol=metric_protocol,
        )
        keys = ["traditional"] + [f"kmeans_r{int(r * 100)}" for r in config.topk_ratios]
        print_summary(m, keys, config.topk_eval, title="MMDocIR Layout - Spherical KMeans")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "mmdocir_layout_kmeans")

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
            metric_protocol=metric_protocol,
        )
        keys = ["traditional"] + [f"rand_r{int(r * 100)}" for r in config.topk_ratios]
        print_summary(m, keys, config.topk_eval, title="MMDocIR Layout - Random")
        save_summary_csv(m, dm, keys, config.topk_eval, config.output_dir, "mmdocir_layout_random")


if __name__ == "__main__":
    main()
