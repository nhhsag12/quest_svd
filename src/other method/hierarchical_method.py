from __future__ import annotations

from common import build_base_parser, finalize_report, load_runtime

from methodology.methods import run_hierarchical


def main():
    parser = build_base_parser("Run Hierarchical Ward pooling baseline only.")
    args = parser.parse_args()

    config, qa_pairs, doc_matrix, doc_mask, query_model, query_processor = load_runtime(args)

    metrics, domain_metrics = run_hierarchical(
        qa_pairs,
        query_processor,
        query_model,
        doc_matrix,
        doc_mask,
        config.topk_eval,
        config.topk_ratios,
        config.device,
    )
    keys = ["traditional"] + [f"hier_r{int(r * 100)}" for r in config.topk_ratios]
    finalize_report(
        metrics,
        domain_metrics,
        keys,
        config.topk_eval,
        config.output_dir,
        title="Hierarchical Ward",
        prefix="hierarchical",
    )


if __name__ == "__main__":
    main()
