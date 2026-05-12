from __future__ import annotations

from common import build_base_parser, finalize_report, load_runtime_with_embeddings

from methodology.methods import run_doc_pooling


def main():
    parser = build_base_parser("Run document 1D/2D pooling baselines only.")
    parser.add_argument(
        "--pooling",
        nargs="+",
        choices=["pool1d", "pool2d"],
        default=["pool1d", "pool2d"],
        help="Pooling strategies to evaluate.",
    )
    parser.add_argument(
        "--pool2d-non-image-tokens",
        type=int,
        default=4,
        help="Fallback number of non-image tokens when exact page images are unavailable.",
    )
    args = parser.parse_args()

    config, qa_pairs, all_page_embeddings, all_page_indices, query_model, query_processor = load_runtime_with_embeddings(args)

    strategies = tuple(dict.fromkeys(args.pooling))
    metrics, domain_metrics = run_doc_pooling(
        qa_pairs,
        query_processor,
        query_model,
        all_page_embeddings,
        all_page_indices,
        config.pages_parquet,
        config.topk_eval,
        config.topk_ratios,
        config.device,
        strategies=strategies,
        pool2d_non_image_tokens=args.pool2d_non_image_tokens,
    )
    keys = ["traditional"] + [f"{strategy}_r{int(r * 100)}" for strategy in strategies for r in config.topk_ratios]
    finalize_report(
        metrics,
        domain_metrics,
        keys,
        config.topk_eval,
        config.output_dir,
        title="Document 1D/2D Pooling",
        prefix="doc_pooling",
    )


if __name__ == "__main__":
    main()
