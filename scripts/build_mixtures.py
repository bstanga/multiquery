#!/usr/bin/env python3
"""
Build set-labeled mixture dataset from cars base aspects. Splits base queries into
train/val/test; samples K in {1,2,3} with Jaccard overlap constraint; writes
datasets/cars_mixtures/ with corpus, queries.jsonl (text, components, k_target), and qrels.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cars_relevance import aspects_with_relevance, load_base_queries, load_corpus

# K distribution (MVP): K=1,2,3 only. ~50% K=1 helps single-intent queries (one active slot).
K_DIST = {1: 0.50, 2: 0.30, 3: 0.15}
K_MAX_MVP = 3

TEMPLATES = {
    1: ["{a}"],
    2: ["{a} or {b}"],
    3: ["{a} or {b} or {c}", "{a}, {b}, or {c}", "either {a}, {b}, or {c}"],
}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def sample_k() -> int:
    r = random.random()
    acc = 0.0
    for k, p in K_DIST.items():
        acc += p
        if r < acc:
            return k
    return 3


def main():
    p = argparse.ArgumentParser(description="Build cars_mixtures from cars base aspects.")
    p.add_argument("--data", type=str, default=None, help="Cars dataset dir (default: datasets/cars)")
    p.add_argument("--out", type=str, default=None, help="Output dir (default: datasets/cars_mixtures)")
    p.add_argument("--train", type=int, default=500, help="Number of train mixtures")
    p.add_argument("--val", type=int, default=100, help="Number of val mixtures")
    p.add_argument("--test", type=int, default=100, help="Number of test mixtures")
    p.add_argument("--jaccard-max", type=float, default=0.2, help="Max Jaccard overlap between aspect rel sets")
    p.add_argument("--min-positives", type=int, default=3, help="Min relevant docs per aspect")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data) if args.data else Path(__file__).resolve().parent.parent / "datasets" / "cars"
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "datasets" / "cars_mixtures"
    random.seed(args.seed)

    corpus = load_corpus(data_dir)
    base_queries = load_base_queries(data_dir)
    aspects = aspects_with_relevance(corpus, base_queries)
    aspects = [(qid, text, rel) for qid, text, rel in aspects if len(rel) >= args.min_positives]
    print(f"Loaded {len(aspects)} base aspects with >= {args.min_positives} positives each")

    # Split by aspect identity (no leakage)
    n = len(aspects)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    random.shuffle(aspects)
    train_aspects = aspects[:n_train]
    val_aspects = aspects[n_train : n_train + n_val]
    test_aspects = aspects[n_train + n_val :]
    print(f"Split aspects: train={len(train_aspects)}, val={len(val_aspects)}, test={len(test_aspects)}")

    def build_mixtures(split_aspects: list, target_count: int, split_name: str) -> tuple[list[dict], list[tuple]]:
        mixtures = []
        qrels_list = []
        max_attempts = target_count * 50
        attempts = 0
        while len(mixtures) < target_count and attempts < max_attempts:
            attempts += 1
            k = sample_k()
            if k > len(split_aspects):
                continue
            chosen = random.sample(split_aspects, k)
            qids, texts, rels = zip(*chosen)
            # Jaccard constraint: all pairs have overlap < jaccard_max
            ok = True
            for i in range(k):
                for j in range(i + 1, k):
                    if jaccard(rels[i], rels[j]) >= args.jaccard_max:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            components = list(texts)
            rel_union = set()
            for r in rels:
                rel_union |= r
            template = random.choice(TEMPLATES[k])
            if k == 1:
                text = template.format(a=components[0])
            elif k == 2:
                text = template.format(a=components[0], b=components[1])
            else:
                text = template.format(a=components[0], b=components[1], c=components[2])
            qid = f"mix_{split_name}_{len(mixtures)}"
            mixtures.append({
                "_id": qid,
                "text": text,
                "components": components,
                "k_target": k,
            })
            for doc_id in rel_union:
                qrels_list.append((qid, doc_id, 1))
        return mixtures, qrels_list

    train_mix, train_qrels = build_mixtures(train_aspects, args.train, "train")
    val_mix, val_qrels = build_mixtures(val_aspects, args.val, "val")
    test_mix, test_qrels = build_mixtures(test_aspects, args.test, "test")
    print(f"Generated: train={len(train_mix)}, val={len(val_mix)}, test={len(test_mix)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qrels").mkdir(exist_ok=True)

    # Corpus: copy from cars
    shutil.copy(data_dir / "corpus.jsonl", out_dir / "corpus.jsonl")

    # Queries: all mixtures (train + val + test) in one file for BEIR; qrels per split
    all_queries = train_mix + val_mix + test_mix
    with open(out_dir / "queries.jsonl", "w") as f:
        for q in all_queries:
            f.write(json.dumps(q) + "\n")

    def write_qrels(qrels: list[tuple[str, str, int]], path: Path):
        with open(path, "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for qid, doc_id, score in qrels:
                f.write(f"{qid}\t{doc_id}\t{score}\n")

    write_qrels(train_qrels, out_dir / "qrels" / "train.tsv")
    write_qrels(val_qrels, out_dir / "qrels" / "val.tsv")
    write_qrels(test_qrels, out_dir / "qrels" / "test.tsv")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
