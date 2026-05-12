from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from methodology.cluster_pooling.spherical_kmeans import spherical_kmeans
from methodology.cluster_pooling.ward import ward_pool_scores_all_ratios
from methodology.doc_pooling import compress_embeddings, infer_grids_from_pages
from metrics.common import hit_metrics, top_k_indices
from metrics.layout import hit_metrics_layout
from metrics.store import record_metric
from methodology.retrieval import build_doc_matrix, fast_maxsim
from methodology.svd.attention_importance import build_content_mask_qwen, compute_svd_importance_softplus
from model.colsmol import encode_query_live


def _evaluate_sample(scores, item, topk_eval, metric_protocol: str = "set"):
    top_idx = top_k_indices(scores, max(topk_eval))
    if metric_protocol == "layout_area":
        return hit_metrics_layout(
            top_idx,
            item.gt_local_indices,
            topk_eval,
            candidate_bbox_list=getattr(item, "candidate_bbox_list", None),
            layout_mapping_raw=getattr(item, "layout_mapping_raw", None),
        )
    return hit_metrics(top_idx, item.gt_local_indices, topk_eval)


def encode_plain_query(question: str, query_processor, query_model, device: str):
    q_text = f"Query: {question}" + "<pad>" * 10
    q_inputs = query_processor(text=[q_text], return_tensors="pt", padding="longest", max_length=600).to(device)
    with torch.no_grad():
        q_proj = query_model(**q_inputs)
    attn_mask = q_inputs["attention_mask"][0]
    idx = torch.where(attn_mask > 0)[0]
    q_emb = q_proj[0][idx].float()
    return F.normalize(q_emb, dim=-1), q_inputs


def run_traditional(qa_pairs, query_processor, query_model, doc_matrix, doc_mask, topk_eval, device: str, metric_protocol: str = "set"):
    metrics = {}
    domain_metrics = {}

    for item in tqdm(qa_pairs, desc="Traditional"):
        q_norm, _ = encode_plain_query(item.question, query_processor, query_model, device)
        doc_mat = doc_matrix[item.doc_embed_indices]
        doc_msk = doc_mask[item.doc_embed_indices]

        scores = fast_maxsim(q_norm, doc_mat, doc_msk).sum(dim=0).cpu().tolist()
        m = _evaluate_sample(scores, item, topk_eval, metric_protocol=metric_protocol)
        record_metric(metrics, domain_metrics, "traditional", m, item.domain, topk_eval)

    return metrics, domain_metrics


def run_hierarchical(qa_pairs, query_processor, query_model, doc_matrix, doc_mask, topk_eval, topk_ratios, device: str, metric_protocol: str = "set"):
    metrics = {}
    domain_metrics = {}

    for item in tqdm(qa_pairs, desc="Hierarchical Ward"):
        q_norm, _ = encode_plain_query(item.question, query_processor, query_model, device)
        doc_mat = doc_matrix[item.doc_embed_indices]
        doc_msk = doc_mask[item.doc_embed_indices]

        base_scores = fast_maxsim(q_norm, doc_mat, doc_msk).sum(dim=0).cpu().tolist()
        m_base = _evaluate_sample(base_scores, item, topk_eval, metric_protocol=metric_protocol)
        record_metric(metrics, domain_metrics, "traditional", m_base, item.domain, topk_eval)

        ratio_scores, _ = ward_pool_scores_all_ratios(q_norm, doc_mat, doc_msk, topk_ratios)
        for r in topk_ratios:
            key = f"hier_r{int(r * 100)}"
            scores = ratio_scores[r].cpu().tolist()
            m = _evaluate_sample(scores, item, topk_eval, metric_protocol=metric_protocol)
            record_metric(metrics, domain_metrics, key, m, item.domain, topk_eval)

    return metrics, domain_metrics


def run_spherical_kmeans(
    qa_pairs,
    query_processor,
    query_model,
    doc_matrix,
    doc_mask,
    topk_eval,
    topk_ratios,
    kmeans_iters: int,
    device: str,
    metric_protocol: str = "set",
):
    metrics = {}
    domain_metrics = {}

    for item in tqdm(qa_pairs, desc="Spherical KMeans"):
        q_norm, _ = encode_plain_query(item.question, query_processor, query_model, device)
        doc_mat = doc_matrix[item.doc_embed_indices]
        doc_msk = doc_mask[item.doc_embed_indices]
        n = q_norm.shape[0]

        base_scores = fast_maxsim(q_norm, doc_mat, doc_msk).sum(dim=0).cpu().tolist()
        m_base = _evaluate_sample(base_scores, item, topk_eval, metric_protocol=metric_protocol)
        record_metric(metrics, domain_metrics, "traditional", m_base, item.domain, topk_eval)

        for r in topk_ratios:
            key = f"kmeans_r{int(r * 100)}"
            k = max(1, int(n * r))
            centroids = spherical_kmeans(q_norm, k, n_iters=kmeans_iters)
            scores = fast_maxsim(centroids, doc_mat, doc_msk).sum(dim=0).cpu().tolist()
            m = _evaluate_sample(scores, item, topk_eval, metric_protocol=metric_protocol)
            record_metric(metrics, domain_metrics, key, m, item.domain, topk_eval)

    return metrics, domain_metrics


def run_random_pruning(
    qa_pairs,
    query_processor,
    query_model,
    doc_matrix,
    doc_mask,
    topk_eval,
    topk_ratios,
    n_random_seeds: int,
    device: str,
    metric_protocol: str = "set",
):
    metrics = {}
    domain_metrics = {}

    for item in tqdm(qa_pairs, desc="Random Pruning"):
        q_norm, _ = encode_plain_query(item.question, query_processor, query_model, device)
        doc_mat = doc_matrix[item.doc_embed_indices]
        doc_msk = doc_mask[item.doc_embed_indices]
        n = q_norm.shape[0]

        base_scores = fast_maxsim(q_norm, doc_mat, doc_msk).sum(dim=0).cpu().tolist()
        m_base = _evaluate_sample(base_scores, item, topk_eval, metric_protocol=metric_protocol)
        record_metric(metrics, domain_metrics, "traditional", m_base, item.domain, topk_eval)

        for r in topk_ratios:
            key = f"rand_r{int(r * 100)}"
            k = max(1, int(n * r))
            acc = torch.zeros(doc_mat.shape[0], device=q_norm.device)
            for _ in range(n_random_seeds):
                idx = torch.randperm(n, device=q_norm.device)[:k]
                q_keep = q_norm[idx]
                acc += fast_maxsim(q_keep, doc_mat, doc_msk).sum(dim=0)
            scores = (acc / n_random_seeds).cpu().tolist()
            m = _evaluate_sample(scores, item, topk_eval, metric_protocol=metric_protocol)
            record_metric(metrics, domain_metrics, key, m, item.domain, topk_eval)

    return metrics, domain_metrics


def run_doc_pooling(
    qa_pairs,
    query_processor,
    query_model,
    all_page_embeddings,
    all_page_indices,
    pages_parquet,
    topk_eval,
    topk_ratios,
    device: str,
    metric_protocol: str = "set",
    strategies: tuple[str, ...] = ("pool1d", "pool2d"),
    pool2d_non_image_tokens: int = 4,
):
    metrics = {}
    domain_metrics = {}

    pooled_matrices = {}
    pooled_masks = {}
    grids = None
    slices = None
    if "pool2d" in strategies:
        grids, slices = infer_grids_from_pages(
            all_page_embeddings,
            pages_parquet,
            all_page_indices,
            non_image_tokens=pool2d_non_image_tokens,
        )

    for strategy in strategies:
        for r in topk_ratios:
            compressed = compress_embeddings(
                all_page_embeddings,
                strategy,
                r,
                grids=grids,
                slices=slices,
                non_image_tokens=pool2d_non_image_tokens,
            )
            mat, mask = build_doc_matrix(compressed, device)
            key = f"{strategy}_r{int(r * 100)}"
            pooled_matrices[key] = mat
            pooled_masks[key] = mask

    base_doc_matrix, base_doc_mask = build_doc_matrix(all_page_embeddings, device)

    for item in tqdm(qa_pairs, desc="Doc 1D/2D Pooling"):
        q_norm, _ = encode_plain_query(item.question, query_processor, query_model, device)

        base_mat = base_doc_matrix[item.doc_embed_indices]
        base_msk = base_doc_mask[item.doc_embed_indices]
        base_scores = fast_maxsim(q_norm, base_mat, base_msk).sum(dim=0).cpu().tolist()
        m_base = _evaluate_sample(base_scores, item, topk_eval, metric_protocol=metric_protocol)
        record_metric(metrics, domain_metrics, "traditional", m_base, item.domain, topk_eval)

        for strategy in strategies:
            for r in topk_ratios:
                key = f"{strategy}_r{int(r * 100)}"
                doc_mat = pooled_matrices[key][item.doc_embed_indices]
                doc_msk = pooled_masks[key][item.doc_embed_indices]
                scores = fast_maxsim(q_norm, doc_mat, doc_msk).sum(dim=0).cpu().tolist()
                m = _evaluate_sample(scores, item, topk_eval, metric_protocol=metric_protocol)
                record_metric(metrics, domain_metrics, key, m, item.domain, topk_eval)

    return metrics, domain_metrics


def run_attention_pruning(
    qa_pairs,
    query_processor,
    query_model,
    doc_matrix,
    doc_mask,
    topk_eval,
    topk_ratios,
    attn_n_layers_list,
    device: str,
    metric_protocol: str = "set",
):
    metrics = {}
    domain_metrics = {}

    for item in tqdm(qa_pairs, desc="Attention Pruning"):
        proj, raw_outputs, q_inputs, _ = encode_query_live(item.question, query_processor, query_model, device)
        attn_mask = q_inputs["attention_mask"][0].float()
        idx = torch.where(attn_mask > 0)[0]
        q_norm = F.normalize(proj[0][idx].float(), dim=-1)

        doc_mat = doc_matrix[item.doc_embed_indices]
        doc_msk = doc_mask[item.doc_embed_indices]

        base_scores = fast_maxsim(q_norm, doc_mat, doc_msk).sum(dim=0).cpu().tolist()
        m_base = _evaluate_sample(base_scores, item, topk_eval, metric_protocol=metric_protocol)
        record_metric(metrics, domain_metrics, "traditional", m_base, item.domain, topk_eval)

        n = idx.numel()
        for n_layers in attn_n_layers_list:
            if raw_outputs.attentions is not None and len(raw_outputs.attentions) >= 1:
                attn_subset = list(raw_outputs.attentions[-n_layers:])
                imp_2d = torch.zeros(1, q_inputs["attention_mask"].shape[1], device=q_norm.device)
                for attn_layer in attn_subset:
                    col_sum = attn_layer.float().sum(dim=2).mean(dim=1)
                    imp_2d += col_sum
                imp = imp_2d[0][idx]
            else:
                imp = torch.ones(n, device=q_norm.device)

            sorted_idx = torch.argsort(imp, descending=True)
            for r in topk_ratios:
                k = max(1, int(n * r))
                keep_idx = sorted_idx[:k]
                imp_keep = imp[keep_idx]
                q_keep = q_norm[keep_idx]

                scores_trad = fast_maxsim(q_keep, doc_mat, doc_msk).sum(dim=0).cpu().tolist()
                key_trad = f"attn_L{n_layers}_r{int(r * 100)}_trad"
                m_trad = _evaluate_sample(scores_trad, item, topk_eval, metric_protocol=metric_protocol)
                record_metric(metrics, domain_metrics, key_trad, m_trad, item.domain, topk_eval)

                imp_keep_n = imp_keep / imp_keep.sum().clamp(min=1e-8)
                scores_weighted = (fast_maxsim(q_keep, doc_mat, doc_msk) * imp_keep_n.unsqueeze(-1)).sum(dim=0).cpu().tolist()
                key_w = f"attn_L{n_layers}_r{int(r * 100)}_weighted"
                m_w = _evaluate_sample(scores_weighted, item, topk_eval, metric_protocol=metric_protocol)
                record_metric(metrics, domain_metrics, key_w, m_w, item.domain, topk_eval)

    return metrics, domain_metrics


def run_ours_svd_cluster_pool(
    qa_pairs,
    query_processor,
    query_model,
    doc_matrix,
    doc_mask,
    topk_eval,
    topk_ratios,
    n_last_layers_list,
    normalize_modes,
    device: str,
    metric_protocol: str = "set",
):
    metrics = {}
    domain_metrics = {}

    for item in tqdm(qa_pairs, desc="Ours SVD + ClusterPool"):
        proj, raw_outputs, q_inputs, _ = encode_query_live(item.question, query_processor, query_model, device)

        attn_mask_1d = q_inputs["attention_mask"][0].float()
        content_mask_1d = build_content_mask_qwen(q_inputs, query_processor)[0].float()
        trad_idx = torch.where(attn_mask_1d > 0)[0]
        method_idx = torch.where(content_mask_1d > 0)[0]
        if method_idx.numel() == 0:
            method_idx = trad_idx

        q_embed = proj[0].float()
        q_method_norm = F.normalize(q_embed[method_idx].float(), dim=-1)

        doc_mat = doc_matrix[item.doc_embed_indices]
        doc_msk = doc_mask[item.doc_embed_indices]

        content_mask_2d = content_mask_1d.unsqueeze(0)
        all_attns = raw_outputs.attentions

        for n_layers in n_last_layers_list:
            if all_attns is not None and len(all_attns) > 0:
                attn_list = list(all_attns[-n_layers:])
                n_actual = len(attn_list)
            else:
                attn_list = []
                n_actual = 0

            layer_weights = torch.exp(torch.linspace(0, 1, max(n_actual, 1), device=q_method_norm.device))
            layer_weights /= layer_weights.sum()

            if n_actual > 0:
                importance = compute_svd_importance_softplus(attn_list, content_mask_2d, layer_weights)
            else:
                importance = content_mask_2d.clone()

            imp_valid = (importance * content_mask_2d)[0][method_idx].float()
            sorted_idx = torch.argsort(imp_valid, descending=True)
            n = method_idx.numel()

            for r in topk_ratios:
                n_keep = max(1, int(n * r))
                keep_idx = sorted_idx[:n_keep]
                disc_idx = sorted_idx[n_keep:]

                q_kept = q_method_norm[keep_idx]
                q_disc = q_method_norm[disc_idx]
                imp_kept = imp_valid[keep_idx]
                imp_disc = imp_valid[disc_idx]

                if q_disc.shape[0] > 0:
                    sim_assign = torch.mm(q_disc, q_kept.t())
                    cluster_ids = sim_assign.argmax(dim=-1)

                    pooled_raw = []
                    pooled_w = []
                    for c in cluster_ids.unique():
                        members = (cluster_ids == c).nonzero(as_tuple=True)[0]
                        w = imp_disc[members]
                        w_sum = w.sum().clamp(min=1e-8)
                        w_norm = w / w_sum
                        pv = (q_disc[members] * w_norm.unsqueeze(-1)).sum(dim=0)
                        pooled_raw.append(pv)
                        pooled_w.append(w_sum)

                    pooled_pre = F.normalize(torch.stack(pooled_raw), dim=-1)
                    pooled_w = torch.stack(pooled_w)
                    all_q_pre = torch.cat([q_kept, pooled_pre], dim=0)
                else:
                    pooled_w = None
                    all_q_pre = q_kept

                m_all = fast_maxsim(all_q_pre, doc_mat, doc_msk)
                m_kept = m_all[:n_keep]
                m_pool = m_all[n_keep:]

                if pooled_w is not None and m_pool.shape[0] > 0:
                    pool_contrib = (m_pool * pooled_w.unsqueeze(-1)).sum(dim=0)
                    pool_contrib = pool_contrib / imp_disc.sum().clamp(min=1e-8)
                else:
                    pool_contrib = torch.zeros(doc_mat.shape[0], device=q_method_norm.device)

                for mode in normalize_modes:
                    tag = f"L{n_layers}_norm{mode}_r{int(r * 100)}"

                    scores_trad = (m_kept.sum(dim=0) + pool_contrib).cpu().tolist()
                    key_trad = f"{tag}_trad"
                    m_trad = _evaluate_sample(scores_trad, item, topk_eval, metric_protocol=metric_protocol)
                    record_metric(metrics, domain_metrics, key_trad, m_trad, item.domain, topk_eval)

                    imp_kept_n = imp_kept / imp_kept.sum().clamp(min=1e-8)
                    scores_w = ((m_kept * imp_kept_n.unsqueeze(-1)).sum(dim=0) + pool_contrib).cpu().tolist()
                    key_w = f"{tag}_weighted"
                    m_w = _evaluate_sample(scores_w, item, topk_eval, metric_protocol=metric_protocol)
                    record_metric(metrics, domain_metrics, key_w, m_w, item.domain, topk_eval)

    return metrics, domain_metrics
