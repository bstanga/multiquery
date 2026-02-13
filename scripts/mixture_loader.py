"""
Load mixture dataset: queries.jsonl (text, components, k_target) + qrels for train/val.
Returns list of dicts with text, components, k_target, positive_doc_ids, query_id.
"""
import json
from pathlib import Path


def load_qrels_union(qrels_path: Path) -> dict[str, list[str]]:
    """qid -> list of positive doc ids."""
    qrels = {}
    with open(qrels_path) as f:
        next(f)
        for line in f:
            qid, doc_id, _ = line.strip().split("\t")
            qrels.setdefault(qid, []).append(doc_id)
    return qrels


def load_mixture_train(data_dir: Path, split: str = "train") -> list[dict]:
    """
    Load mixture examples for a split. Each dict has:
    - query_id, text, components (list of K strings), k_target (int), positive_doc_ids (list)
    """
    data_dir = Path(data_dir)
    qrels = load_qrels_union(data_dir / "qrels" / f"{split}.tsv")
    examples = []
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["_id"]
            if not qid.startswith(f"mix_{split}_"):
                continue
            text = obj["text"]
            components = obj["components"]
            k_target = obj["k_target"]
            pos = qrels.get(qid, [])
            examples.append({
                "query_id": qid,
                "text": text,
                "components": components,
                "k_target": k_target,
                "positive_doc_ids": pos,
            })
    return examples


def load_mixture_val(data_dir: Path) -> list[dict]:
    return load_mixture_train(data_dir, split="val")
