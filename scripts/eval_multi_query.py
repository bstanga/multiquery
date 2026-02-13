#!/usr/bin/env python3
"""
Evaluate Multi-Slot retrieval using Weighted Max-Similarity Fusion.
Scores each doc as max_k( p_k * sim(slot_k, doc) ), utilizing the model's
predicted activation scores (p) to suppress hallucinated/inactive slots.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from model_multi_slot import MultiSlotQwenQueryEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cars"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "multi_slot"
BATCH_SIZE = 32
K_VALUES = [1, 3, 5, 10, 100]


def load_model(ckpt_path: Path = None):
    if ckpt_path is None:
        ckpt_path = CKPT_DIR / "multi_slot.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Run train_multi_slot.py first. Missing {ckpt_path}")
    
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    # Load MultiSlot model
    model = MultiSlotQwenQueryEncoder(k_max=ckpt.get("k_max", 8))
    model.attn.load_state_dict(ckpt["attn"])
    model.proj.load_state_dict(ckpt["proj"])
    model.act.load_state_dict(ckpt["act"])
    model.slots = torch.nn.Parameter(ckpt["slots"].float())
    
    backbone_device = next(model.backbone.parameters()).device
    model.attn = model.attn.to(backbone_device)
    model.proj = model.proj.to(backbone_device)
    model.act = model.act.to(backbone_device)
    model.slots.data = model.slots.data.to(backbone_device)
    model.eval()
    return model


def retrieve_weighted_max_sim(model: MultiSlotQwenQueryEncoder, corpus: dict, queries: dict, top_k: int = 100):
    """
    Weighted Max-Similarity Fusion:
    Score = max_k ( p_k * sim(slot_k, doc) )
     Complexity: O(Q * K * D) where K is K_max.
    """
    print("Encoding corpus...")
    corpus_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id] for doc_id in corpus_ids]
    doc_texts = []
    for doc in corpus_list:
        title = doc.get("title", "")
        body = doc.get("text", "")
        doc_texts.append(f"{title}\n{body}".strip() if title else body)
        
    encode_device = next(model.backbone.parameters()).device
    doc_emb = model.encode_corpus_backbone(doc_texts, encode_device, batch_size=BATCH_SIZE)
    doc_emb = doc_emb.cpu().numpy()  # (num_docs, dim)

    query_ids = list(queries.keys())
    query_texts = [queries[qid] if isinstance(queries[qid], str) else queries[qid]["text"] for qid in query_ids]
    results = {qid: {} for qid in query_ids}

    print(f"Retrieving for {len(query_texts)} queries...")
    for i in range(0, len(query_texts), BATCH_SIZE):
        batch_texts = query_texts[i : i + BATCH_SIZE]
        batch_ids = query_ids[i : i + BATCH_SIZE]
        
        # Get embeddings (B, K, D) and probabilities (B, K)
        q_emb, p_out = model.encode_texts(batch_texts, encode_device)
        q_emb = q_emb.detach().cpu().numpy()
        p_out = p_out.detach().cpu().numpy()
        
        for b in range(q_emb.shape[0]):
            # 1. Get all slot scores: (K, num_docs)
            all_slot_scores = np.dot(q_emb[b], doc_emb.T)
            
            # 2. Apply Activation Scores (Weighted Max-Sim)
            # p_b: (K, 1)
            p_b = p_out[b][:, np.newaxis]
            weighted_scores = all_slot_scores * p_b
            
            # 3. Reduce to single score per doc: max over slots
            final_scores = np.max(weighted_scores, axis=0)
            
            # 4. Rank
            top_idx = np.argsort(final_scores)[::-1][:top_k]
            for rank, idx in enumerate(top_idx):
                results[batch_ids[b]][corpus_ids[idx]] = float(final_scores[idx])
                
    return results


def filter_or_only(queries: dict, qrels: dict) -> tuple[dict, dict]:
    """Keep only query IDs starting with 'or_' and their qrels."""
    or_qids = {qid for qid in queries if str(qid).startswith("or_")}
    queries_or = {qid: queries[qid] for qid in or_qids}
    qrels_or = {qid: qrels[qid] for qid in or_qids if qid in qrels}
    return queries_or, qrels_or


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hard", action="store_true", help="Use hard test set")
    p.add_argument("--or-only", action="store_true", help="Evaluate on OR queries only (from test set)")
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    args = p.parse_args()
    
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    model = load_model(ckpt_path)
    
    if args.hard:
        hard_path = DATA_DIR / "qrels" / "test_hard.tsv"
        loader = GenericDataLoader(data_folder=str(DATA_DIR), qrels_file=str(hard_path))
        corpus, queries, qrels = loader.load_custom()
        print("Evaluating Multi-Slot (Weighted Max-Sim) on hard set...")
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=str(DATA_DIR)).load(split="test")
        if args.or_only:
            queries, qrels = filter_or_only(queries, qrels)
            if not queries:
                raise SystemExit("No OR queries in test set.")
            print(f"Evaluating Multi-Slot (Weighted Max-Sim) on OR-only ({len(queries)} queries)...")
        else:
            print("Evaluating Multi-Slot (Weighted Max-Sim) on full test set...")
    
    top_k = max(K_VALUES)
    results = retrieve_weighted_max_sim(model, corpus, queries, top_k=top_k)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, K_VALUES)
    print("nDCG@10:", ndcg["NDCG@10"])
    print("Recall@100:", recall["Recall@100"])


if __name__ == "__main__":
    main()
