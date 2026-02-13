#!/usr/bin/env python3
"""
Fine-tune the Qwen embedding model to output 2 vectors per query for OR/composite queries.
Supervision: decomposition loss — the two output vectors should match the embeddings
of the two sub-queries (left and right of " or "), from the frozen backbone.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_multi import MultiQueryEncoder, NUM_VECTORS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cars"
OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "multi_query"
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 3
SEED = 42


def load_or_train_pairs(data_dir: Path) -> list[tuple[str, str, str]]:
    """Load (composite_query, sub1, sub2) for OR queries that appear in train qrels."""
    train_qrels_path = data_dir / "qrels" / "train.tsv"
    train_query_ids = set()
    with open(train_qrels_path) as f:
        next(f)  # header
        for line in f:
            qid, _, _ = line.strip().split("\t")
            train_query_ids.add(qid)
    pairs = []
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["_id"]
            if not qid.startswith("or_") or qid not in train_query_ids:
                continue
            text = obj["text"]
            if " or " not in text:
                continue
            parts = text.split(" or ", 1)
            if len(parts) != 2:
                continue
            sub1, sub2 = parts[0].strip(), parts[1].strip()
            if not sub1 or not sub2:
                continue
            pairs.append((text, sub1, sub2))
    return pairs


def main():
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_or_train_pairs(DATA_DIR)
    print(f"Loaded {len(pairs)} OR training pairs from {DATA_DIR}")
    if not pairs:
        raise SystemExit("No OR queries in train set. Run generate_cars_dataset.py and ensure train has OR qrels.")

    model = MultiQueryEncoder(num_vectors=NUM_VECTORS)
    # Freeze backbone; only train multi_head
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()
    model.multi_head = model.multi_head.to(DEVICE)
    model.backbone = model.backbone.to(DEVICE)
    opt = torch.optim.AdamW(model.multi_head.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            composite = [p[0] for p in batch]
            sub1 = [p[1] for p in batch]
            sub2 = [p[2] for p in batch]

            # Targets from frozen backbone (single vector per sub-query)
            backbone_device = next(model.backbone.parameters()).device
            with torch.no_grad():
                t1 = model.get_single_vector_backbone(sub1, backbone_device)
                t2 = model.get_single_vector_backbone(sub2, backbone_device)
                t1 = F.normalize(t1.float(), p=2, dim=-1).to(DEVICE)
                t2 = F.normalize(t2.float(), p=2, dim=-1).to(DEVICE)

            # Forward composite through backbone + multi_head
            from base import format_query
            formatted = [format_query(q) for q in composite]
            enc = model.tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(backbone_device) for k, v in enc.items()}
            out = model.backbone(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            pooled = model._pool_last_token(hidden, enc["attention_mask"])
            pooled = pooled.to(DEVICE)
            v = model.multi_head(pooled.float())  # (B, 2, dim)
            v1, v2 = v[:, 0, :], v[:, 1, :]

            loss1 = (1 - F.cosine_similarity(v1, t1, dim=-1)).mean()
            loss2 = (1 - F.cosine_similarity(v2, t2, dim=-1)).mean()
            loss = loss1 + loss2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / n_batches
        print(f"Epoch {epoch+1}/{EPOCHS}  avg_loss={avg:.4f}")

    # Save multi-head and config (eval loads same backbone by name)
    save_path = OUT_DIR / "multi_head.pt"
    torch.save({
        "state_dict": model.multi_head.state_dict(),
        "hidden_size": model.hidden_size,
        "num_vectors": model.num_vectors,
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
    }, save_path)
    print(f"Saved multi-head to {save_path}")
    print("Run: python scripts/eval_multi_query.py  # to evaluate 2-vector retrieval")


if __name__ == "__main__":
    main()
