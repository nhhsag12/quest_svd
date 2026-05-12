from __future__ import annotations

from common import build_base_parser, finalize_report, load_runtime

from methodology.methods import run_traditional


def main():
    parser = build_base_parser("Run Traditional MaxSim baseline only.")
    args = parser.parse_args()

    config, qa_pairs, doc_matrix, doc_mask, query_model, query_processor = load_runtime(args)

    metrics, domain_metrics = run_traditional(
        qa_pairs,
        query_processor,
        query_model,
        doc_matrix,
        doc_mask,
        config.topk_eval,
        config.device,
    )
    finalize_report(
        metrics,
        domain_metrics,
        ["traditional"],
        config.topk_eval,
        config.output_dir,
        title="Traditional MaxSim",
        prefix="traditional",
    )


if __name__ == "__main__":
    main()
