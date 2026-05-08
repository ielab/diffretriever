#!/usr/bin/env python3
"""
Evaluate all retrieval modes from a single all_steps encoding run.

Corpus is processed SHARD-BY-SHARD: only one shard is in memory at a time.
Queries are loaded once (small). A running top-k is maintained per query
with global doc offsets, so total RAM is O(shards × shard_size × H) → O(shard).

All representations come from repr_hidden [N, K, H]:
  - Dense (single_dense): mean-pool repr_hidden → [N, H], dot product
  - ColBERT (multi_dense): MaxSim over repr_hidden [N, K, H]
  - When K=1, dense and colbert are identical.

Retrieval modes:
  single_dense              : mean(repr_hidden) [N, H] — single-vector dense
  multi_dense               : repr_hidden [N, K, H] — ColBERT MaxSim
  sparse_max                : max-pool sparse across K → dot product
  fusion_single_sparse_max  : normalized_fusion(single_dense, sparse_max)
  fusion_multi_sparse_max   : normalized_fusion(multi_dense, sparse_max)

Outputs (all written to --output_dir):
  summary.json          — all modes, average metrics only
  {mode}.json           — per-query metrics + average at bottom
  {mode}.trec           — TREC run file (qid Q0 docid rank score tag)

Usage:
    python scripts/evaluate_sweep.py \\
        --query_dir  embeddings/msmarco/dream_few_k4_s1/queries-dev \\
        --corpus_dir embeddings/msmarco/dream_few_k4_s1/corpus \\
        --qrels      data/msmarco/qrels.dev.tsv \\
        --output_dir results/msmarco/dream_few_k4_s1 \\
        --top_k 100
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query loading  (queries are small — load all at once)
# ---------------------------------------------------------------------------

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from shard_io import load_shard as _load_shard_auto, list_shards as _list_shards


def load_query_data(data_dir: str, only_shard: int = -1) -> dict:
    """Load query embeddings from a single file or set of shards.

    If `only_shard >= 0`, load just that shard (for parallelizing eval over
    query splits — each job processes 1/N of the queries).  Falls back to the
    full multi-shard load when only_shard < 0 (default).
    """
    # only_shard mode: must be a multi-shard layout.
    if only_shard >= 0:
        for ext in ('.safetensors', '.safetensors.zst', '.pt'):
            p = Path(data_dir) / f'all_steps_shard_{only_shard}{ext}'
            if p.exists():
                logger.info(f"Loading query shard {only_shard}: {p}")
                d = _load_shard_auto(p)
                if 'repr_hidden' not in d and 'quotation_emb' in d:
                    d['repr_hidden'] = d['quotation_emb'].unsqueeze(1)
                return d
        raise FileNotFoundError(
            f"No all_steps_shard_{only_shard}.* in {data_dir} "
            f"(use scripts/split_query_embeddings.py to back-fill)"
        )

    # Single-file layout: all_steps_embeddings.{safetensors,safetensors.zst,pt}
    for ext in ('.safetensors', '.safetensors.zst', '.pt'):
        single = Path(data_dir) / f'all_steps_embeddings{ext}'
        if single.exists():
            logger.info(f"Loading query file: {single}")
            d = _load_shard_auto(single)
            if 'repr_hidden' not in d and 'quotation_emb' in d:
                d['repr_hidden'] = d['quotation_emb'].unsqueeze(1)
            return d

    # Multi-shard query layout
    shard_bases = _list_shards(data_dir, prefix='all_steps_shard_')
    if not shard_bases:
        raise FileNotFoundError(f"No all_steps files in {data_dir}")

    logger.info(f"Loading {len(shard_bases)} query shards from {data_dir}...")
    ids_all, repr_all, si_all, sv_all = [], [], [], []
    for sb in shard_bases:
        d = _load_shard_auto(sb)
        ids_all.extend(d.get('ids', []))
        if 'repr_hidden' in d:
            repr_all.append(d['repr_hidden'])
        elif 'quotation_emb' in d:
            repr_all.append(d['quotation_emb'].unsqueeze(1))
        if 'sparse_indices' in d:
            si_all.append(d['sparse_indices'])
            sv_all.append(d['sparse_values'])
    out = {'ids': ids_all}
    if repr_all:
        out['repr_hidden'] = torch.cat(repr_all, 0)
    if si_all:
        out['sparse_indices'] = torch.cat(si_all, 0)
        out['sparse_values']  = torch.cat(sv_all, 0)
    return out


def get_corpus_shards(corpus_dir: str):
    """Return sorted list of corpus shard base paths (auto-detects .pt/.safetensors/.zst)."""
    shard_bases = _list_shards(corpus_dir, prefix='all_steps_shard_')
    if shard_bases:
        return shard_bases
    for ext in ('.safetensors', '.safetensors.zst', '.pt'):
        single = Path(corpus_dir) / f'all_steps_embeddings{ext}'
        if single.exists():
            # Strip the extension(s); load_shard re-probes all variants from the base.
            base_name = single.name[: -len(ext)]
            return [single.with_name(base_name)]
    raise FileNotFoundError(f"No all_steps corpus files in {corpus_dir}")


def _peek_corpus_kp(first_shard_base) -> int:
    """Return the corpus K (dim-1 of repr_hidden) from a shard's safetensors
    header — no tensor data decoded, just the metadata JSON.

    Lets the eval loop know K_p ahead of time so ColBERT/multi_dense is
    correctly enabled for cross-K runs where K_q=1 but K_p>1.  Returns 1 if
    the header can't be read or repr_hidden isn't present (fall-back to
    legacy behaviour).
    """
    import struct as _struct
    st_path  = Path(str(first_shard_base) + '.safetensors')
    zst_path = Path(str(first_shard_base) + '.safetensors.zst')

    def _header_from(reader):
        hl_raw = reader.read(8)
        if len(hl_raw) != 8:
            return None
        hl = _struct.unpack('<Q', hl_raw)[0]
        hb = reader.read(hl)
        if len(hb) != hl:
            return None
        try:
            return json.loads(hb)
        except Exception:
            return None

    header = None
    try:
        if st_path.exists():
            with open(st_path, 'rb') as f:
                header = _header_from(f)
        elif zst_path.exists():
            try:
                import zstandard as _z
            except ImportError:
                return 1
            with open(zst_path, 'rb') as f:
                with _z.ZstdDecompressor().stream_reader(f) as reader:
                    header = _header_from(reader)
    except Exception:
        return 1

    if not isinstance(header, dict):
        return 1
    for name in ('repr_hidden', 'quotation_emb'):
        meta = header.get(name)
        if isinstance(meta, dict):
            shape = meta.get('shape')
            if isinstance(shape, list) and len(shape) >= 2:
                try:
                    return int(shape[1]) if name == 'repr_hidden' else 1
                except (TypeError, ValueError):
                    continue
    return 1


def load_qrels(qrels_file: str):
    """Load qrels as {qid: {docid: rel}} with ALL relevance levels.

    Do NOT filter by threshold here — pass the full dict to pytrec_eval
    (which handles relevance_level internally) or to compute_metrics.
    """
    qrels = defaultdict(dict)
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, did, rel = parts
            elif len(parts) == 3:
                qid, did, rel = parts
            else:
                continue
            try:
                qrels[qid][did] = int(rel)
            except ValueError:
                continue  # skip TSV header lines (e.g. BEIR "query-id corpus-id score")
    return qrels


# ---------------------------------------------------------------------------
# Running top-k accumulator
# ---------------------------------------------------------------------------

def merge_hits(global_mode: dict, shard_hits: dict, shard_offset: int, top_k: int):
    """Merge per-shard top-k results into global top-k with adjusted global doc indices."""
    for qidx, hits in shard_hits.items():
        existing = global_mode.get(qidx, [])
        for score, local_idx in hits:
            existing.append((score, shard_offset + local_idx))
        existing.sort(key=lambda x: -x[0])
        global_mode[qidx] = existing[:top_k]


# ---------------------------------------------------------------------------
# Per-shard scoring
# ---------------------------------------------------------------------------

def _topk_to_dict(topk_vals: torch.Tensor, topk_idx: torch.Tensor,
                  offset: int = 0) -> dict:
    """Convert [N, k] topk tensors to {row: [(val, idx), ...]} dict (batch CPU transfer)."""
    tv = topk_vals.cpu()
    ti = topk_idx.cpu()
    N = tv.shape[0]
    results = {}
    tv_list = tv.tolist()
    ti_list = ti.tolist()
    for j in range(N):
        results[offset + j] = list(zip(tv_list[j], ti_list[j]))
    return results


def dense_scores(q_emb: torch.Tensor, c_emb: torch.Tensor,
                 top_k: int, chunk_size: int = 512,
                 device: torch.device = None) -> dict:
    """Dense dot-product scores. Returns {qidx: [(score, local_doc_idx)]}."""
    N_q = q_emb.shape[0]
    N_c = c_emb.shape[0]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c_dev = c_emb if (c_emb.device == device and c_emb.dtype == torch.bfloat16) \
            else c_emb.to(device=device, dtype=torch.bfloat16)
    results = {}
    k = min(top_k, N_c)
    for i in range(0, N_q, chunk_size):
        q_chunk = q_emb[i:i + chunk_size].to(device=device, dtype=torch.bfloat16)
        scores = (q_chunk @ c_dev.T).float()
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        results.update(_topk_to_dict(topk_vals, topk_idx, i))
    del c_dev
    return results


def compute_diversity_weights(q_repr: torch.Tensor) -> torch.Tensor:
    """Per-position diversity weights for query multi-vectors (CPU, run once).

    w_0 = 1
    w_k = clamp(1 - max_{j<k} cos_sim(q_k, q_j), min=0)   k >= 1

    Intuition: a position that is highly similar to an earlier one carries
    redundant information and should be down-weighted in aggregation.

    Args:
        q_repr: [N, K, H] L2-normalised query vectors
    Returns:
        weights: [N, K] non-negative, NOT normalised (use w/w.sum() for mean)
    """
    N, K, H = q_repr.shape
    if K == 1:
        return torch.ones(N, 1)
    w = torch.ones(N, K)
    for k in range(1, K):
        # max cosine sim of position k to any earlier position
        sim_to_prev = (q_repr[:, k:k + 1, :] * q_repr[:, :k, :]).sum(dim=-1)  # [N, k]
        max_sim = sim_to_prev.max(dim=-1).values                                # [N]
        w[:, k] = (1.0 - max_sim).clamp(min=0.0)
    return w


def colbert_scores(q_emb: torch.Tensor, c_emb: torch.Tensor,
                   top_k: int, chunk_size: int = 64,
                   q_weights: torch.Tensor = None,
                   device: torch.device = None) -> dict:
    """ColBERT MaxSim: Score(q,d) = sum_i w_i * max_j (q_i · d_j).

    q_emb:     [N_q, K, H]  (L2-normalised)
    c_emb:     [N_c, K, H]  (L2-normalised)
    q_weights: [N_q, K] optional per-position weights (diversity or uniform)
    """
    N_q, K_q, H = q_emb.shape
    N_c, K_c, _ = c_emb.shape
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w_dev  = q_weights.to(device) if q_weights is not None else None

    # Ensure corpus is on GPU in bfloat16
    if c_emb.device == device and c_emb.dtype == torch.bfloat16:
        c_flat = c_emb.view(N_c * K_c, H)
    else:
        c_flat = c_emb.view(N_c * K_c, H).to(device=device, dtype=torch.bfloat16)

    # Score matrix peak: [B*K_q, N_c*K_c] in bf16 = B*K_q*N_c*K_c*2 bytes
    # On 96GB H100, allow up to ~16GB for the score matrix → ~8B elements in bf16
    MAX_BF16_ELEMENTS = 8_000_000_000
    q_chunk_size = max(1, min(chunk_size, MAX_BF16_ELEMENTS // max(1, K_q * N_c * K_c)))

    results = {}
    for i in range(0, N_q, q_chunk_size):
        q_chunk = q_emb[i:i + q_chunk_size]
        if q_chunk.device != device or q_chunk.dtype != torch.bfloat16:
            q_chunk = q_chunk.to(device=device, dtype=torch.bfloat16)
        B = q_chunk.shape[0]
        q_flat = q_chunk.reshape(B * K_q, H)
        # bf16 matmul → max over K_c in bf16 (avoids large float32 4D tensor)
        scores_flat = q_flat @ c_flat.T                          # [B*K_q, N_c*K_c] bf16
        maxsim = scores_flat.view(B * K_q, N_c, K_c).max(dim=-1).values  # [B*K_q, N_c] bf16
        # Clamp to >= 0: matches the implicit 0-floor that zero-padded K positions
        # used to provide before we started trimming trailing zero slots.
        # Idempotent on untrimmed shards (max was already >= 0 from zero pad).
        maxsim = maxsim.clamp(min=0).view(B, K_q, N_c).float()  # [B, K_q, N_c] float32 (small)
        del scores_flat
        if w_dev is not None:
            w = w_dev[i:i + q_chunk_size].unsqueeze(-1)
            colbert = (maxsim * w).sum(dim=1)
        else:
            colbert = maxsim.sum(dim=1)                          # [B, N_c]
        del maxsim
        k = min(top_k, N_c)
        topk_vals, topk_idx = colbert.topk(k, dim=-1)
        results.update(_topk_to_dict(topk_vals, topk_idx, i))
    del c_flat
    return results


def _build_compact_sparse(si: torch.Tensor, sv: torch.Tensor,
                           mapping: torch.Tensor, M: int,
                           device: torch.device = None):
    """Build max-pooled compact sparse matrix.

    si: [N, K, topk]  (int indices)
    sv: [N, K, topk]  (float values, non-negative)
    mapping: [vocab_size] — maps vocab_id to compact col index (−1 = not present)

    Returns:
        pooled_max: tensor [N, M] — max across all K positions
    """
    N, K, topk = si.shape
    vocab_sz = len(mapping)
    if device is None:
        device = si.device

    # Move everything to GPU and vectorize across K (no Python loop)
    si_dev = si.to(device=device, dtype=torch.long)
    sv_dev = sv.to(device=device, dtype=torch.float32)
    mapping_dev = mapping.to(device=device) if mapping.device != device else mapping

    in_rng = (si_dev >= 0) & (si_dev < vocab_sz)
    compact = mapping_dev[si_dev.clamp(0, vocab_sz - 1)]  # [N, K, topk]
    valid = (compact >= 0) & in_rng

    # Flatten: each (n, k, t) position contributes to pooled_max[n, compact[n,k,t]]
    n_idx = torch.arange(N, device=device).view(N, 1, 1).expand(N, K, topk)
    flat_idx = (n_idx * M + compact).reshape(-1)
    flat_vals = sv_dev.reshape(-1)
    ok = valid.reshape(-1)

    pooled_max = torch.zeros(N * M, dtype=torch.float32, device=device)
    if ok.any():
        pooled_max.scatter_reduce_(0, flat_idx[ok], flat_vals[ok],
                                    reduce='amax', include_self=True)
    return pooled_max.view(N, M)


def precompute_query_sparse(q_si: torch.Tensor, q_sv: torch.Tensor,
                             device: torch.device = None):
    """Build max-pooled query compact sparse.

    q_si/q_sv: [N_q, K, topk]
    Returns:
        q_sp_max: tensor [N_q, M_q] — max-pooled across positions
        mapping:  [vocab_size] long, −1 for terms not in queries (on device)
        M_q:      number of unique query terms
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = int(q_si.max().item()) + 1
    q_terms    = q_si.reshape(-1).unique().sort().values
    M_q        = len(q_terms)
    mapping    = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    mapping[q_terms.long().to(device)] = torch.arange(M_q, device=device)
    q_sp_max = _build_compact_sparse(q_si, q_sv, mapping, M_q, device=device)
    return q_sp_max, mapping, M_q


def sparse_scores_shard(q_sp_max: torch.Tensor,
                         q_mapping: torch.Tensor, M_q: int,
                         c_si: torch.Tensor, c_sv: torch.Tensor,
                         top_k: int, chunk_size: int = 4096,
                         device: torch.device = None):
    """Compute max-pool sparse scores: max-pooled query × max-pooled corpus.

    Returns: {qidx: [(score, local_doc_idx)]}
    """
    N_q = q_sp_max.shape[0]
    N_c = c_si.shape[0]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_max_dev = q_sp_max.to(device=device, dtype=torch.bfloat16)

    all_vals, all_idx = [], []

    for i in range(0, N_c, chunk_size):
        c_si_c = c_si[i:i + chunk_size]
        c_sv_c = c_sv[i:i + chunk_size]
        B = c_si_c.shape[0]

        c_pooled = _build_compact_sparse(c_si_c, c_sv_c, q_mapping, M_q, device=device)
        scores   = (q_max_dev @ c_pooled.to(torch.bfloat16).T).float()

        kk = min(top_k, B)
        mv, mi = scores.topk(kk, dim=-1)
        all_vals.append(mv.cpu())
        all_idx.append((mi + i).cpu())

    all_v = torch.cat(all_vals, dim=1)
    all_i = torch.cat(all_idx,  dim=1)
    kk = min(top_k, all_v.shape[1])
    _, top_pos = all_v.topk(kk, dim=-1)
    fv = all_v.gather(1, top_pos)
    fi = all_i.gather(1, top_pos)
    return _topk_to_dict(fv, fi)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def normalized_fusion(results_a: dict, results_b: dict, top_k: int, alpha: float = 0.5) -> dict:
    """PromptReps-style fusion: min-max normalize each run then linearly combine (alpha=0.5)."""
    all_qidxs = set(results_a) | set(results_b)
    fused = {}
    for qidx in all_qidxs:
        hits_a = results_a.get(qidx, [])
        hits_b = results_b.get(qidx, [])

        def normalize(hits):
            if not hits:
                return {}
            scores = [s for s, _ in hits]
            min_s, max_s = min(scores), max(scores)
            denom = max(max_s - min_s, 1e-9)
            return {doc: (s - min_s) / denom for s, doc in hits}

        norm_a = normalize(hits_a)
        norm_b = normalize(hits_b)
        combined = {}
        for doc in set(norm_a) | set(norm_b):
            combined[doc] = alpha * norm_a.get(doc, 0.0) + (1 - alpha) * norm_b.get(doc, 0.0)
        fused[qidx] = [(s, d) for d, s in sorted(combined.items(), key=lambda x: -x[1])[:top_k]]
    return fused


# ---------------------------------------------------------------------------
# Evaluation & output
# ---------------------------------------------------------------------------

def compute_metrics(retrieval_results: dict, q_ids, c_ids, qrels, top_k: int = 10,
                    rel_threshold: int = 1):
    """Evaluate retrieval results.

    Uses pytrec_eval when available (gold standard, handles graded NDCG,
    relevance_level / -l, and judged-queries-only averaging).
    Falls back to a custom graded NDCG implementation otherwise.
    """
    try:
        import pytrec_eval as _pte

        # Build run: {qid: {docid: score}} — only queries present in qrels
        run = {}
        for qidx, hits in retrieval_results.items():
            qid = q_ids[qidx]
            if qid in qrels:
                run[qid] = {c_ids[doc_idx]: float(score) for score, doc_idx in hits[:top_k]}

        if not run:
            return {f'ndcg@{top_k}': 0.0, f'mrr@{top_k}': 0.0, 'num_queries': 0}, {}

        evaluator = _pte.RelevanceEvaluator(
            dict(qrels),
            {f'ndcg_cut.{top_k}', 'recip_rank'},
            relevance_level=rel_threshold,
        )
        results = evaluator.evaluate(run)

        per_query, ndcg_list, mrr_list = {}, [], []
        ndcg_key = f'ndcg_cut_{top_k}'
        for qid, m in results.items():
            ndcg = m.get(ndcg_key, 0.0)
            mrr  = m.get('recip_rank', 0.0)
            per_query[qid] = {f'ndcg@{top_k}': round(ndcg, 4), f'mrr@{top_k}': round(mrr, 4)}
            ndcg_list.append(ndcg)
            mrr_list.append(mrr)

        avg = {
            f'ndcg@{top_k}': round(sum(ndcg_list) / len(ndcg_list), 4) if ndcg_list else 0.0,
            f'mrr@{top_k}':  round(sum(mrr_list)  / len(mrr_list),  4) if mrr_list  else 0.0,
            'num_queries': len(ndcg_list),
        }
        return avg, per_query

    except ImportError:
        logger.warning("pytrec_eval not found — using custom graded NDCG fallback. "
                       "Install with: pip install pytrec-eval-terrier")

    # ---- fallback: custom graded NDCG ----
    per_query = {}
    ndcg_list, mrr_list = [], []

    for qidx, hits in retrieval_results.items():
        qid = q_ids[qidx]
        rel = qrels.get(qid, {})
        # skip queries with no judged-relevant docs (mirrors trec_eval default / -c behaviour)
        if not any(r >= rel_threshold for r in rel.values()):
            continue

        dcg = sum(
            (2 ** rel.get(c_ids[doc_idx], 0) - 1) / math.log2(rank + 2)
            for rank, (_, doc_idx) in enumerate(hits[:top_k])
        )
        idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in
                   enumerate(sorted(rel.values(), reverse=True)[:top_k]))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        mrr = next(
            (1.0 / (rank + 1) for rank, (_, doc_idx) in enumerate(hits[:top_k])
             if rel.get(c_ids[doc_idx], 0) >= rel_threshold),
            0.0,
        )

        per_query[qid] = {f'ndcg@{top_k}': round(ndcg, 4), f'mrr@{top_k}': round(mrr, 4)}
        ndcg_list.append(ndcg)
        mrr_list.append(mrr)

    if not ndcg_list:
        avg = {f'ndcg@{top_k}': 0.0, f'mrr@{top_k}': 0.0, 'num_queries': 0}
    else:
        avg = {
            f'ndcg@{top_k}': round(sum(ndcg_list) / len(ndcg_list), 4),
            f'mrr@{top_k}':  round(sum(mrr_list)  / len(mrr_list),  4),
            'num_queries': len(ndcg_list),
        }
    return avg, per_query


def write_trec(retrieval_results: dict, q_ids, c_ids, filepath: str, tag: str):
    with open(filepath, 'w') as f:
        for qidx in sorted(retrieval_results):
            qid = q_ids[qidx]
            for rank, (score, doc_idx) in enumerate(retrieval_results[qidx]):
                f.write(f"{qid} Q0 {c_ids[doc_idx]} {rank + 1} {score:.6f} {tag}\n")


def save_mode(retrieval_results, q_ids, c_ids, qrels,
              output_dir: str, mode: str, eval_top_k: int, rel_threshold: int = 1):
    avg, per_query = compute_metrics(retrieval_results, q_ids, c_ids, qrels, eval_top_k, rel_threshold)
    with open(os.path.join(output_dir, f'{mode}.json'), 'w') as f:
        json.dump({'per_query': per_query, 'average': avg}, f, indent=2)
    write_trec(retrieval_results, q_ids, c_ids,
               os.path.join(output_dir, f'{mode}.trec'), tag=mode)
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_dir',  required=True)
    parser.add_argument('--corpus_dir', required=True)
    parser.add_argument('--qrels',      required=True)
    parser.add_argument('--output_dir', required=True,
                        help='Directory for summary.json, per-mode .json and .trec')
    parser.add_argument('--top_k',      type=int, default=1000,
                        help='Candidates retrieved per query')
    parser.add_argument('--eval_top_k', type=int, default=10,
                        help='NDCG/MRR cutoff (default 10)')
    parser.add_argument('--rel_threshold', type=int, default=1,
                        help='Minimum relevance to count as relevant (default 1). '
                             'Use 2 for TREC DL19/DL20 (trec_eval -l 2).')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Query chunk size for dense matmul')
    parser.add_argument('--sparse_chunk', type=int, default=4096,
                        help='Corpus chunk size for sparse scoring')
    parser.add_argument('--profile', action='store_true',
                        help='Time each step per shard and report breakdown at end')
    parser.add_argument('--profile_max_shards', type=int, default=20,
                        help='When --profile, stop after this many shards (default: 20)')
    parser.add_argument('--query_shard', type=int, default=-1,
                        help='Load only this query shard from query_dir (parallelize '
                             'eval across query splits). Default -1 = load all.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ queries
    logger.info("Loading query embeddings...")
    q_data = load_query_data(args.query_dir, only_shard=args.query_shard)
    q_ids  = q_data['ids']
    logger.info(f"  {len(q_ids)} queries (before qrels filter)")

    # Load qrels early so we can filter queries to only those with relevance judgments.
    # datasets like HotpotQA ship queries.jsonl with all splits (97K) but qrels only
    # covers the dev/test split (~7K) — no point scoring the rest.
    logger.info("Loading qrels...")
    qrels = load_qrels(args.qrels)

    keep_ids = set(qrels.keys())
    if len(keep_ids) < len(q_ids):
        keep_idx = [i for i, qid in enumerate(q_ids) if qid in keep_ids]
        keep_t   = torch.tensor(keep_idx, dtype=torch.long)
        q_ids = [q_ids[i] for i in keep_idx]
        for key in ('repr_hidden', 'sparse_indices', 'sparse_values'):
            if key in q_data:
                q_data[key] = q_data[key][keep_t]
        logger.info(f"  Filtered to {len(q_ids)} queries with qrels")

    N_q = len(q_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    has_repr      = 'repr_hidden'    in q_data
    has_sparse    = 'sparse_indices' in q_data

    # repr_hidden [N, K, H]
    # single_dense: mean-pool across K → [N, H]  (always)
    # multi_dense:  ColBERT MaxSim over K vectors (K>1 only)
    K = q_data['repr_hidden'].shape[1] if has_repr else 0
    if has_repr:
        # Chunked in-place L2-normalize on the bf16 tensor (mirrors the corpus
        # path further down).  The previous F.normalize(.float()) version
        # tripled CPU memory: msmarco-train at K_q=16 is 66 GB bf16, so the
        # fp32 cast + normalize buffer peaked at ~330 GB and OOM-killed the
        # job on the 256 GB cap.  torch.linalg.vector_norm accumulates in
        # fp32 internally even on bf16 inputs, so accuracy is preserved.
        qh = q_data['repr_hidden']
        if qh.dtype != torch.bfloat16:
            qh = qh.to(torch.bfloat16)
        H = qh.shape[2]
        Q_CHUNK = 16384
        _eps = 1e-12
        q_dense = torch.empty(N_q, H, dtype=torch.bfloat16)
        for _s in range(0, N_q, Q_CHUNK):
            _e = min(_s + Q_CHUNK, N_q)
            _block = qh[_s:_e]                                       # bf16 view
            _n = torch.linalg.vector_norm(_block, dim=-1, keepdim=True).clamp(min=_eps)
            _block.div_(_n)                                          # in-place
            # Masked mean-pool: exclude all-zero positions (AR stop-at-quote pads).
            # Sum reduction in fp32 to avoid bf16 drift over K vectors.
            _mask  = (_block.pow(2).sum(dim=-1) > 0).to(torch.float32)
            _count = _mask.sum(dim=1, keepdim=True).clamp(min=1)
            q_dense[_s:_e] = ((_block.float() * _mask.unsqueeze(-1)).sum(dim=1) / _count).to(torch.bfloat16)
            del _block, _n, _mask, _count

        # GPU upload only when q_repr fits comfortably.  Beyond that, keep on
        # CPU; colbert_scores already streams query chunks per shard
        # (`q_chunk.to(device=...)` at L301) so the only cost is one extra
        # H2D copy per shard, dwarfed by the ColBERT matmul.
        q_bytes = N_q * K * H * 2
        if torch.cuda.is_available() and q_bytes < 20 * (1 << 30):
            q_repr = qh.to(device=device)
        else:
            q_repr = qh
    else:
        q_repr  = None
        q_dense = None
    # Peek at the corpus's K_p so cross-K runs (K_q=1, K_p>1) correctly
    # enable ColBERT.  multi_dense with K_q=1 is NOT the same as
    # single_dense: it max-picks over K_p passage tokens rather than
    # mean-pooling them.  When both sides are 1 the two are equivalent
    # and colbert is skipped (just waste).
    _first_shard_paths = get_corpus_shards(args.corpus_dir)
    _K_p = _peek_corpus_kp(_first_shard_paths[0]) if _first_shard_paths else 1
    enable_colbert = has_repr and (K > 1 or _K_p > 1)
    _mode_tag = 'single_dense'
    if enable_colbert:
        _mode_tag += '+multi_dense'
    logger.info(f"  K_q={K}  K_p={_K_p}  modes={_mode_tag}")

    q_sp_max = q_mapping = M_q = None
    if has_sparse:
        logger.info("Precomputing query sparse compact (once)...")
        q_sp_max, q_mapping, M_q = precompute_query_sparse(
            q_data['sparse_indices'], q_data['sparse_values'], device=device
        )
        logger.info(f"  Query sparse compact: {M_q} unique terms, K={K} positions")

    # ------------------------------------------------------------------ corpus shards
    shard_paths = get_corpus_shards(args.corpus_dir)
    logger.info(f"Processing {len(shard_paths)} corpus shards one-by-one...")

    # Running top-k accumulators: {qidx: [(score, global_doc_idx), ...]}
    # single_dense: mean-pool dense (always)
    # multi_dense:  ColBERT MaxSim (K>1 only)
    global_dense   = {} if has_repr           else None
    global_colbert = {} if enable_colbert     else None
    global_sparse  = {} if has_sparse          else None
    c_ids_all         = []   # collect doc IDs as we go (strings, negligible memory)

    pin_memory = os.environ.get("EVAL_PIN_MEMORY", "0") == "1"

    def _load_shard(path):
        """Load shard in a background thread.

        Pinning is opt-in because large shards spend seconds in pin_memory()
        while H2D copy itself is usually sub-second; it also increases
        transient RAM pressure. Set EVAL_PIN_MEMORY=1 to restore pinned H2D.
        """
        d = _load_shard_auto(path)
        for k in ('repr_hidden', 'quotation_emb'):
            if k in d:
                t = d[k]
                if t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
                d[k] = t.contiguous().pin_memory() if pin_memory else t.contiguous()
        return d

    # Profile accumulators: sum of each phase's time across shards
    import time
    prof = {'load_wait': 0.0, 'to_gpu': 0.0, 'normalize_mean': 0.0,
            'dense_score': 0.0, 'colbert_score': 0.0, 'sparse_score': 0.0,
            'merge_hits': 0.0, 'shard_total': 0.0}
    shard_limit = args.profile_max_shards if args.profile else len(shard_paths)
    shard_limit = min(shard_limit, len(shard_paths))

    # Dedicated CUDA stream for H2D copies — lets next shard upload overlap
    # with current shard's compute on the default stream.
    copy_stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None

    # Prefetch window — multiple shards loading in parallel stays ahead of GPU processing.
    # Empirically N=4 is optimal on compute nodes; N=1 is the low-memory
    # setting for tight login-node cgroups (<8 GB RAM).  Override with the
    # EVAL_PREFETCH env var.  Each extra slot costs ~one decompressed
    # shard's worth of CPU RAM (~0.5-3 GB depending on K).
    N_PREFETCH = int(os.environ.get("EVAL_PREFETCH", "4"))
    if N_PREFETCH < 1:
        N_PREFETCH = 1
    with ThreadPoolExecutor(max_workers=N_PREFETCH) as loader:
        futures = {}
        for j in range(min(N_PREFETCH, shard_limit)):
            futures[j] = loader.submit(_load_shard, shard_paths[j])

        for idx in tqdm(range(shard_limit), desc='Corpus shards'):
            t_shard = time.time()

            t0 = time.time()
            shard = futures.pop(idx).result()
            prof['load_wait'] += time.time() - t0

            # Kick off next prefetch to maintain the window
            next_idx = idx + N_PREFETCH
            if next_idx < shard_limit:
                futures[next_idx] = loader.submit(_load_shard, shard_paths[next_idx])

            offset = len(c_ids_all)
            c_ids_all.extend(shard['ids'])

            shard_has_repr   = 'repr_hidden'    in shard
            shard_has_quot   = 'quotation_emb'  in shard  # backward compat
            shard_has_sparse = 'sparse_indices' in shard

            # H2D on dedicated copy stream (pinned CPU tensor from loader thread).
            # Compute stream will wait_stream below before touching c_repr_gpu.
            t0 = time.time()
            if copy_stream is not None:
                with torch.cuda.stream(copy_stream):
                    if shard_has_repr:
                        c_repr_gpu = shard['repr_hidden'].to(
                            device=device, non_blocking=pin_memory)
                    elif shard_has_quot:
                        c_repr_gpu = shard['quotation_emb'].unsqueeze(1).to(
                            device=device, non_blocking=pin_memory)
                    else:
                        c_repr_gpu = None
                # Make compute stream wait for the H2D to finish
                torch.cuda.current_stream(device).wait_stream(copy_stream)
            else:
                if shard_has_repr:
                    c_repr_gpu = shard['repr_hidden'].to(device=device)
                elif shard_has_quot:
                    c_repr_gpu = shard['quotation_emb'].unsqueeze(1).to(device=device)
                else:
                    c_repr_gpu = None
            if args.profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            prof['to_gpu'] += time.time() - t0

            if has_repr and c_repr_gpu is not None:
                t0 = time.time()
                # In-place chunked L2-normalize along last dim.  Doing it as a
                # single F.normalize(c_repr_gpu) allocates a second tensor of
                # the same shape as c_repr_gpu (a ~[N_c, K_p, H] fp16 tensor).
                # At K_p=16, H=4096, N_c≈327k that's ~43 GB → CUDA OOM.
                NORM_CHUNK = 16384
                _eps = 1e-12
                for _s in range(0, c_repr_gpu.shape[0], NORM_CHUNK):
                    _e = min(_s + NORM_CHUNK, c_repr_gpu.shape[0])
                    _n = torch.linalg.vector_norm(
                        c_repr_gpu[_s:_e], dim=-1, keepdim=True
                    ).clamp(min=_eps)
                    c_repr_gpu[_s:_e].div_(_n)
                    del _n

                # single_dense: mean-pool across K → dot product.
                # Done chunked over N_c so the fp32 materialisation of
                # [chunk, K, H] stays small: at K=16, H=4096 the full
                # [N_c, 16, 4096] fp32 tensor is ~28 GB/shard and OOMs.
                _c_mask  = ((c_repr_gpu ** 2).sum(dim=-1) > 0).float()
                _c_count = _c_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [N_c, 1]
                N_c_shard = c_repr_gpu.shape[0]
                H = c_repr_gpu.shape[-1]
                c_dense = torch.empty(
                    N_c_shard, H, dtype=torch.bfloat16, device=c_repr_gpu.device
                )
                MEAN_POOL_CHUNK = 16384
                for _s in range(0, N_c_shard, MEAN_POOL_CHUNK):
                    _e = min(_s + MEAN_POOL_CHUNK, N_c_shard)
                    _sub = c_repr_gpu[_s:_e].float()                     # [B, K, H]
                    _m   = _c_mask[_s:_e].unsqueeze(-1)                  # [B, K, 1]
                    _cnt = _c_count[_s:_e]                                # [B, 1]
                    c_dense[_s:_e] = ((_sub * _m).sum(dim=1) / _cnt).to(torch.bfloat16)
                    del _sub
                if args.profile and torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof['normalize_mean'] += time.time() - t0

                t0 = time.time()
                hits = dense_scores(q_dense, c_dense, args.top_k, args.chunk_size,
                                    device=device)
                if args.profile and torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof['dense_score'] += time.time() - t0

                t0 = time.time()
                merge_hits(global_dense, hits, offset, args.top_k)
                prof['merge_hits'] += time.time() - t0

                if enable_colbert:
                    t0 = time.time()
                    # multi_dense: ColBERT MaxSim over [N_c, K_p, H].
                    # Valid whenever K_q > 1 OR K_p > 1.  When K_q=1,
                    # MaxSim reduces to max-over-passage-tokens — still
                    # distinct from single_dense (mean-over-passage).
                    hits = colbert_scores(q_repr, c_repr_gpu, args.top_k,
                                          chunk_size=min(64, args.chunk_size),
                                          device=device)
                    if args.profile and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    prof['colbert_score'] += time.time() - t0

                    t0 = time.time()
                    merge_hits(global_colbert, hits, offset, args.top_k)
                    prof['merge_hits'] += time.time() - t0
                del c_repr_gpu

            if has_sparse and shard_has_sparse:
                t0 = time.time()
                hits = sparse_scores_shard(
                    q_sp_max, q_mapping, M_q,
                    shard['sparse_indices'], shard['sparse_values'],
                    args.top_k, args.sparse_chunk, device=device)
                if args.profile and torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof['sparse_score'] += time.time() - t0

                t0 = time.time()
                merge_hits(global_sparse, hits, offset, args.top_k)
                prof['merge_hits'] += time.time() - t0

            del shard  # release shard memory before next shard arrives
            prof['shard_total'] += time.time() - t_shard

    logger.info(f"Processed {len(c_ids_all)} corpus docs total")

    if args.profile:
        n = shard_limit
        logger.info(f"\n=== Profile ({n} shards) ===")
        logger.info(f"  Total: {prof['shard_total']:.1f}s  ({prof['shard_total']/n:.2f}s/shard)")
        for key in ('load_wait', 'to_gpu', 'normalize_mean',
                    'dense_score', 'colbert_score', 'sparse_score', 'merge_hits'):
            t = prof[key]
            pct = 100 * t / prof['shard_total'] if prof['shard_total'] > 0 else 0
            logger.info(f"    {key:<16}: {t:6.1f}s  ({t/n:.2f}s/shard, {pct:4.1f}%)")
        logger.info("=============================")
        logger.info(f"Stopping after {n} shards (--profile mode)")
        return

    # ------------------------------------------------------------------ fusions
    global_fuse_single = None
    global_fuse_multi  = None
    if global_sparse is not None:
        if global_dense is not None:
            global_fuse_single = normalized_fusion(global_dense,   global_sparse, args.top_k)
        if global_colbert is not None:
            global_fuse_multi  = normalized_fusion(global_colbert, global_sparse, args.top_k)

    # ------------------------------------------------------------------ evaluate & save
    summary = {}
    modes = [
        ('single_dense',             global_dense),
        ('multi_dense',              global_colbert),      # None for K=1
        ('sparse_max',               global_sparse),
        ('fusion_single_sparse_max', global_fuse_single),
        ('fusion_multi_sparse_max',  global_fuse_multi),   # None for K=1
    ]
    for mode_name, results in modes:
        if results is None:
            continue
        logger.info(f"Evaluating & saving: {mode_name}...")
        avg = save_mode(results, q_ids, c_ids_all, qrels,
                        args.output_dir, mode_name, args.eval_top_k, args.rel_threshold)
        summary[mode_name] = avg
        logger.info(f"  {mode_name}: {avg}")

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary → {summary_path}")

    print(f"\n{'Config':<35} {'NDCG@10':>10} {'MRR@10':>10} {'#Q':>8}")
    print('-' * 67)
    for key, m in summary.items():
        ndcg = m.get(f'ndcg@{args.eval_top_k}', 0)
        mrr  = m.get(f'mrr@{args.eval_top_k}',  0)
        nq   = m.get('num_queries', 0)
        print(f"{key:<35} {ndcg:>10.4f} {mrr:>10.4f} {nq:>8}")


if __name__ == '__main__':
    main()
