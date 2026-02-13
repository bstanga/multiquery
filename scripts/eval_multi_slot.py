#!/usr/bin/env python3
"""
Evaluate multi-slot retrieval: for each query use active slots (by p) and score each doc
as max over active slot–doc cosine. Loads checkpoint from checkpoints/multi_slot/multi_slot.pt.
Use --data datasets/cars_mixtures for mixture test set; --by-k reports by k_target (K=1/2/3).
"""
import argparse
import json
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
MIXTURES_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cars_mixtures"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "multi_slot"
BATCH_SIZE = 32
K_VALUES = [1, 3, 5, 10, 100]


def load_query_k_target(data_dir: Path) -> dict[str, int]:
    """Load query_id -> k_target from mixture queries.jsonl (only mix_* lines)."""
    out = {}
    path = data_dir / "queries.jsonl"
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("_id", "")
            if qid.startswith("mix_"):
                out[qid] = obj.get("k_target", 1)
    return out


def load_model(ckpt_path: Path) -> tuple:
    """Load model and return (model, ckpt_dict) so caller can read trained_on_mixtures etc."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Run train_multi_slot.py first. Missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = MultiSlotQwenQueryEncoder(k_max=ckpt["k_max"])
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
    return model, ckpt


def get_active_slot_indices(p: np.ndarray, p_threshold: float = 0.0, min_slots: int = 2) -> np.ndarray:
    """For one query, return indices of active slots (p > threshold or top min_slots by p)."""
    active = np.where(p > p_threshold)[0]
    if len(active) < min_slots:
        active = np.argsort(p)[::-1][:min_slots]
    if len(active) == 0:
        active = np.array([0])
    return active


def retrieve_slot(
    model: MultiSlotQwenQueryEncoder,
    corpus: dict,
    queries: dict,
    top_k: int = 100,
    p_threshold: float = 0.0,
    min_slots: int = 2,
) -> dict:
    """Score each doc as max over active-slot cosine similarities."""
    corpus_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id] for doc_id in corpus_ids]
    doc_texts = []
    for doc in corpus_list:
        title = doc.get("title", "")
        body = doc.get("text", "")
        doc_texts.append(f"{title}\n{body}".strip() if title else body)
    encode_device = next(model.backbone.parameters()).device
    doc_emb = model.encode_corpus_backbone(doc_texts, encode_device, batch_size=BATCH_SIZE)
    doc_emb = doc_emb.detach().cpu().float().numpy()

    query_ids = list(queries.keys())
    query_texts = [queries[qid] if isinstance(queries[qid], str) else queries[qid]["text"] for qid in query_ids]
    results = {qid: {} for qid in query_ids}

    for i in range(0, len(query_texts), BATCH_SIZE):
        batch_texts = query_texts[i : i + BATCH_SIZE]
        batch_ids = query_ids[i : i + BATCH_SIZE]
        emb, p = model.encode_texts(batch_texts, encode_device)
        emb = emb.detach().cpu().float().numpy()
        p = p.detach().cpu().numpy()
        for b in range(emb.shape[0]):
            slot_emb = emb[b]
            slot_p = p[b]
            active_idx = get_active_slot_indices(slot_p, p_threshold=p_threshold, min_slots=min_slots)
            sims = np.dot(slot_emb[active_idx], doc_emb.T)
            scores = np.max(sims, axis=0)
            top_idx = np.argsort(scores)[::-1][:top_k]
            for rank, idx in enumerate(top_idx):
                results[batch_ids[b]][corpus_ids[idx]] = float(scores[idx])
    return results


def filter_or_only(queries: dict, qrels: dict) -> tuple[dict, dict]:
    or_qids = {qid for qid in queries if str(qid).startswith("or_")}
    return {qid: queries[qid] for qid in or_qids}, {qid: qrels[qid] for qid in or_qids if qid in qrels}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="Path to multi_slot.pt (default: checkpoints/multi_slot/multi_slot.pt)")
    p.add_argument("--data", type=str, default=None, help="Dataset dir (default: datasets/cars). Use datasets/cars_mixtures for mixture test set.")
    p.add_argument("--hard", action="store_true", help="Use hard test set")
    p.add_argument("--or-only", action="store_true", help="Evaluate on OR queries only")
    p.add_argument("--by-k", action="store_true", help="Report metrics by k_target (K=1, K=2, K=3). Requires mixture-style queries.jsonl with k_target.")
    p.add_argument("--p-threshold", type=float, default=0.0, help="Min activation for a slot to be active")
    p.add_argument("--min-slots", type=int, default=1, help="Use at least this many slots (top by p)")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIR / "multi_slot.pt"
    model, ckpt = load_model(ckpt_path)
    trained_on_mixtures = ckpt.get("trained_on_mixtures", False)

    if args.data is not None:
        data_dir = Path(args.data)
    elif trained_on_mixtures:
        data_dir = MIXTURES_DIR
        print("Checkpoint was trained on mixtures; using mixture test set (datasets/cars_mixtures).")
    else:
        data_dir = DATA_DIR

    use_by_k = args.by_k or (trained_on_mixtures and "cars_mixtures" in str(data_dir))

    if args.hard:
        loader = GenericDataLoader(data_folder=str(data_dir), qrels_file=str(data_dir / "qrels" / "test_hard.tsv"))
        corpus, queries, qrels = loader.load_custom()
        print("Evaluating multi-slot on hard set...")
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=str(data_dir)).load(split="test")
        if args.or_only:
            queries, qrels = filter_or_only(queries, qrels)
            if not queries:
                raise SystemExit("No OR queries in test set.")
            print(f"Evaluating multi-slot on OR-only ({len(queries)} queries)...")
        else:
            print(f"Evaluating multi-slot on test set ({len(queries)} queries)...")

    results = retrieve_slot(
        model, corpus, queries,
        top_k=max(K_VALUES),
        p_threshold=args.p_threshold,
        min_slots=args.min_slots,
    )

    if use_by_k:
        query_to_k = load_query_k_target(data_dir)
        if not query_to_k:
            print("Per-K breakdown: no mix_* queries with k_target in queries.jsonl; skipping.")
        else:
            for k_val in [1, 2, 3]:
                qids_k = [qid for qid in queries if query_to_k.get(qid) == k_val]
                if not qids_k:
                    continue
                qrels_k = {qid: qrels[qid] for qid in qids_k if qid in qrels}
                results_k = {qid: results[qid] for qid in qids_k if qid in results}
                if not qrels_k:
                    continue
                ndcg_k, _map_k, recall_k, _prec_k = EvaluateRetrieval.evaluate(qrels_k, results_k, K_VALUES)
                print(f"  K={k_val} (n={len(qids_k)}): nDCG@10={ndcg_k['NDCG@10']:.4f}, Recall@100={recall_k['Recall@100']:.4f}")
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, K_VALUES)
    print("nDCG@10:", ndcg["NDCG@10"])
    print("Recall@100:", recall["Recall@100"])


if __name__ == "__main__":
    main()
