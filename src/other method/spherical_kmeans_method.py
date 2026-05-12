from __future__ import annotations

from common import build_base_parser, finalize_report, load_runtime

from methodology.methods import run_spherical_kmeans


def main():
    parser = build_base_parser("Run Spherical KMeans baseline only.")
    parser.add_argument("--kmeans-iters", type=int, default=10)
    args = parser.parse_args()

    config, qa_pairs, doc_matrix, doc_mask, query_model, query_processor = load_runtime(args)

    metrics, domain_metrics = run_spherical_kmeans(
        qa_pairs,
        query_processor,
        query_model,
        doc_matrix,
        doc_mask,
        config.topk_eval,
        config.topk_ratios,
        config.kmeans_iters,
        config.device,
    )
    keys = ["traditional"] + [f"kmeans_r{int(r * 100)}" for r in config.topk_ratios]
    finalize_report(
        metrics,
        domain_metrics,
        keys,
        config.topk_eval,
        config.output_dir,
        title="Spherical KMeans",
        prefix="spherical_kmeans",
    )


if __name__ == "__main__":
    main()
