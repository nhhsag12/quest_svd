from __future__ import annotations

from common import build_base_parser, finalize_report, load_runtime

from methodology.methods import run_random_pruning


def main():
    parser = build_base_parser("Run Random pruning baseline only.")
    parser.add_argument("--n-random-seeds", type=int, default=1)
    args = parser.parse_args()

    config, qa_pairs, doc_matrix, doc_mask, query_model, query_processor = load_runtime(args)

    metrics, domain_metrics = run_random_pruning(
        qa_pairs,
        query_processor,
        query_model,
        doc_matrix,
        doc_mask,
        config.topk_eval,
        config.topk_ratios,
        config.n_random_seeds,
        config.device,
    )
    keys = ["traditional"] + [f"rand_r{int(r * 100)}" for r in config.topk_ratios]
    finalize_report(
        metrics,
        domain_metrics,
        keys,
        config.topk_eval,
        config.output_dir,
        title="Random Pruning",
        prefix="random_pruning",
    )


if __name__ == "__main__":
    main()
