import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from collections import defaultdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluation pipeline for dense retrieval."""
    
    @staticmethod
    def load_qrels(qrels_file: Union[str, Path]) -> Dict[str, Dict[str, int]]:
        """Load qrels file (TREC format or TSV, space or tab delimited)."""
        qrels = defaultdict(dict)
        with open(qrels_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Try tab first, then whitespace (covers TREC space-delimited format)
                parts = line.split('	')
                if len(parts) < 3:
                    parts = line.split()
                if len(parts) == 4:  # TREC format: qid 0 docid rel
                    qid, _, docid, rel = parts
                elif len(parts) == 3:  # TSV: qid docid rel
                    qid, docid, rel = parts
                else:
                    continue
                qrels[str(qid)][str(docid)] = int(rel)
        return dict(qrels)

    @staticmethod
    def compute_metrics(
        rankings: Dict[str, List[str]], 
        qrels: Dict[str, Dict[str, int]], 
        k_values: List[int] = [1, 5, 10, 20, 100]
    ) -> Dict[str, float]:
        """Compute standard IR metrics (MRR, Recall, NDCG)."""
        metrics = {}
        
        for k in k_values:
            # MRR@k
            mrr_scores = []
            for qid, ranked_docs in rankings.items():
                if qid not in qrels:
                    continue
                for rank, docid in enumerate(ranked_docs[:k], 1):
                    if str(docid) in qrels[qid] and qrels[qid][str(docid)] > 0:
                        mrr_scores.append(1.0 / rank)
                        break
                else:
                    mrr_scores.append(0.0)
            metrics[f'MRR@{k}'] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
            
            # Recall@k
            recall_scores = []
            for qid, ranked_docs in rankings.items():
                if qid not in qrels:
                    continue
                relevant = set(d for d, r in qrels[qid].items() if r > 0)
                if not relevant:
                    continue
                retrieved = set(str(d) for d in ranked_docs[:k])
                recall_scores.append(len(relevant & retrieved) / len(relevant))
            metrics[f'Recall@{k}'] = float(np.mean(recall_scores)) if recall_scores else 0.0
            
            # NDCG@k
            ndcg_scores = []
            for qid, ranked_docs in rankings.items():
                if qid not in qrels:
                    continue
                dcg = 0.0
                for rank, docid in enumerate(ranked_docs[:k], 1):
                    rel = qrels[qid].get(str(docid), 0)
                    dcg += (2 ** rel - 1) / np.log2(rank + 1)
                
                ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
                idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rels))
                ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
            metrics[f'NDCG@{k}'] = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        
        return metrics

    def evaluate(
        self,
        query_embeddings: torch.Tensor,
        corpus_embeddings: torch.Tensor,
        qrels_file: Union[str, Path],
        query_ids: List[str],
        corpus_ids: List[str],
        top_k: int = 100,
        batch_size: int = 128,
    ) -> Dict[str, float]:
        """Complete evaluation pipeline."""
        qrels = self.load_qrels(qrels_file)
        
        logger.info(f"Evaluating {len(query_ids)} queries against {len(corpus_ids)} documents...")
        
        rankings = {}
        for i in range(0, len(query_ids), batch_size):
            batch_q = query_embeddings[i:i + batch_size].to(corpus_embeddings.device)
            scores = torch.mm(batch_q, corpus_embeddings.t())
            
            actual_k = min(top_k, scores.size(1))
            top_k_scores, top_k_indices = scores.topk(actual_k, dim=1)
            
            for j, qid in enumerate(query_ids[i:i + batch_size]):
                rankings[str(qid)] = [str(corpus_ids[idx]) for idx in top_k_indices[j].tolist()]
                
        return self.compute_metrics(rankings, qrels)
