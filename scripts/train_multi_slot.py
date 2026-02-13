#!/usr/bin/env python3
"""
Train the multi-slot query encoder on OR pairs or on mixture data (--mixtures).

OR mode: two slots match sub-query targets; diversity + activation losses.
Mixture mode: retrieval loss (max-slot score vs in-batch negatives), count loss
(MSE sum(p) vs k_target), assignment loss (Hungarian match slots to components).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import format_query
from model_multi_slot import MultiSlotQwenQueryEncoder, diversity_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cars"
MIXTURES_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cars_mixtures"
OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "multi_slot"
SEED = 42
K_MAX = 8
DIVERSITY_MARGIN = 0.2
DIVERSITY_WEIGHT = 0.1
ACT_WEIGHT = 0.2
# Mixture loss weights
RETRIEVAL_WEIGHT = 1.0
COUNT_WEIGHT = 0.5
ASSIGN_WEIGHT = 0.5
UNMATCHED_P_WEIGHT = 0.2
MATCHED_P_WEIGHT = 0.1   # encourage matched slots' p to be high (clean K=1)
P_BIN_WEIGHT = 0.05      # encourage p near 0/1
RETRIEVAL_TEMPERATURE = 0.07


def load_or_train_pairs(data_dir: Path) -> list[tuple[str, str, str]]:
    """Load (composite_query, sub1, sub2) for OR queries in train qrels."""
    train_qrels_path = data_dir / "qrels" / "train.tsv"
    train_query_ids = set()
    with open(train_qrels_path) as f:
        next(f)
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


def _load_corpus(data_dir: Path) -> dict[str, str]:
    """doc_id -> text (from corpus.jsonl)."""
    out = {}
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            out[obj["_id"]] = obj.get("text", obj.get("title", ""))
    return out


def _encode_docs_backbone(model: MultiSlotQwenQueryEncoder, doc_ids: list[str], corpus: dict[str, str], device: torch.device, batch_size: int = 64) -> tuple[dict[str, int], torch.Tensor]:
    """Encode docs with backbone; return id2idx and (N, D) tensor."""
    texts = [corpus[did] for did in doc_ids]
    emb = model.encode_corpus_backbone(texts, device, batch_size=batch_size)
    id2idx = {did: i for i, did in enumerate(doc_ids)}
    return id2idx, emb.float()


def _component_embeddings_in_slot_space(
    model: MultiSlotQwenQueryEncoder,
    flat_components: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Backbone encode then proj+normalize so targets are in slot space. (N, D)."""
    backbone_device = next(model.backbone.parameters()).device
    out_list = []
    for i in range(0, len(flat_components), batch_size):
        batch = flat_components[i : i + batch_size]
        h = model.get_single_vector_backbone(batch, backbone_device)
        h = h.to(device)
        proj = model.proj(h)
        out_list.append(F.normalize(proj, p=2, dim=-1))
    return torch.cat(out_list, dim=0)


def train_mixtures(args, data_dir: Path, model, backbone_device, opt, scaler, use_amp: bool):
    from mixture_loader import load_mixture_train

    train_examples = load_mixture_train(data_dir, split="train")
    corpus = _load_corpus(data_dir)
    train_examples = [
        ex for ex in train_examples
        if ex["positive_doc_ids"] and any(p in corpus for p in ex["positive_doc_ids"])
    ]
    print(f"Loaded {len(train_examples)} mixture examples from {data_dir} (with positives in corpus)")
    if not train_examples:
        raise SystemExit("No mixture train examples with positives in corpus. Run scripts/build_mixtures.py first.")
    batch_size = args.batch_size
    accum_steps = args.accum_steps
    epochs = args.epochs

    for epoch in range(epochs):
        # Cache component embeddings for this epoch (avoids re-encoding every batch)
        unique_components = list({c for ex in train_examples for c in ex["components"]})
        with torch.no_grad():
            aspect_cache_tensor = _component_embeddings_in_slot_space(model, unique_components, DEVICE)
        aspect_cache = {text: aspect_cache_tensor[i].to(DEVICE) for i, text in enumerate(unique_components)}

        model.train()
        random.shuffle(train_examples)
        total_loss = 0.0
        n_batches = 0
        opt.zero_grad()
        for i in range(0, len(train_examples), batch_size):
            batch = train_examples[i : i + batch_size]
            texts = [ex["text"] for ex in batch]
            k_targets = torch.tensor([ex["k_target"] for ex in batch], dtype=torch.long, device=DEVICE)

            formatted = [format_query(q) for q in texts]
            enc = model.tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(backbone_device) for k, v in enc.items()}

            def forward_and_loss():
                emb, p_out = model(enc["input_ids"], enc["attention_mask"])
                emb = emb.to(DEVICE)
                p_out = p_out.to(DEVICE)
                B = emb.size(0)

                # --- Retrieval loss: true in-batch negatives (one pos doc per query, BxB scores, cross_entropy) ---
                pos_doc_ids = [random.choice([p for p in ex["positive_doc_ids"] if p in corpus]) for ex in batch]
                pos_doc_texts = [corpus[pid] for pid in pos_doc_ids]
                with torch.no_grad():
                    d = model.encode_corpus_backbone(pos_doc_texts, backbone_device, batch_size=B)
                    d = F.normalize(d.float().to(DEVICE), p=2, dim=-1)
                # score[b, j] = max_k (emb[b,k] · d[j]); gate by p but blend in ungated to avoid vanishing gradients early
                sim = torch.einsum("bkd,jd->bkj", emb, d)
                sim = sim * p_out.unsqueeze(-1) + 0.1 * sim
                scores = sim.max(dim=1).values / RETRIEVAL_TEMPERATURE
                labels = torch.arange(B, device=DEVICE)
                retrieval_loss = F.cross_entropy(scores, labels)

                # --- Count loss ---
                count_loss = (p_out.sum(dim=1) - k_targets.float()).pow(2).mean()

                # --- Assignment loss (Hungarian) + unmatched p; use cached component embeddings ---
                flat_components = [c for ex in batch for c in ex["components"]]
                offsets = [0]
                for ex in batch:
                    offsets.append(offsets[-1] + len(ex["components"]))
                if flat_components:
                    aspect_raw = torch.stack([aspect_cache[c] for c in flat_components])
                else:
                    aspect_raw = torch.zeros(0, emb.size(-1), device=DEVICE)
                assign_loss = torch.tensor(0.0, device=DEVICE)
                unmatched_p_loss = torch.tensor(0.0, device=DEVICE)
                matched_p_loss = torch.tensor(0.0, device=DEVICE)
                n_assign = 0
                n_b_processed = 0
                for b in range(B):
                    k_t = batch[b]["k_target"]
                    slot_b = emb[b]
                    aspect_b = aspect_raw[offsets[b] : offsets[b + 1]]
                    aspect_b = aspect_b[:k_t]
                    if aspect_b.size(0) == 0:
                        continue
                    n_b_processed += 1
                    cost = 1.0 - torch.mm(aspect_b, slot_b.T)
                    if cost.size(0) > model.k_max:
                        cost = cost[: model.k_max, :]
                    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
                    matched_slots = list(col_ind)
                    for ri, cj in zip(row_ind, col_ind):
                        assign_loss = assign_loss + (1.0 - F.cosine_similarity(emb[b : b + 1, cj], aspect_b[ri : ri + 1], dim=-1)).squeeze()
                        n_assign += 1
                    if matched_slots:
                        matched_p_loss = matched_p_loss + (1 - p_out[b, matched_slots].mean())
                    unmatched = [j for j in range(model.k_max) if j not in matched_slots]
                    if unmatched:
                        unmatched_p_loss = unmatched_p_loss + p_out[b, unmatched].mean()
                if n_assign > 0:
                    assign_loss = assign_loss / n_assign
                if n_b_processed > 0 and unmatched_p_loss.requires_grad:
                    unmatched_p_loss = unmatched_p_loss / n_b_processed
                if n_b_processed > 0 and matched_p_loss.requires_grad:
                    matched_p_loss = matched_p_loss / n_b_processed

                div_loss = diversity_loss(emb, margin=DIVERSITY_MARGIN)
                p_bin = (p_out * (1 - p_out)).mean()
                loss = (
                    RETRIEVAL_WEIGHT * retrieval_loss
                    + COUNT_WEIGHT * count_loss
                    + ASSIGN_WEIGHT * assign_loss
                    + UNMATCHED_P_WEIGHT * unmatched_p_loss
                    + MATCHED_P_WEIGHT * matched_p_loss
                    + DIVERSITY_WEIGHT * div_loss
                    + P_BIN_WEIGHT * p_bin
                )
                return loss

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = forward_and_loss() / accum_steps
                scaler.scale(loss).backward()
            else:
                loss = forward_and_loss() / accum_steps
                loss.backward()

            total_loss += loss.item() * accum_steps
            n_batches += 1

            if n_batches % accum_steps == 0:
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()

        if n_batches % accum_steps != 0:
            if use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
        print(f"Epoch {epoch+1}/{epochs}  avg_loss={total_loss / max(1, n_batches):.4f}")


def main():
    p = argparse.ArgumentParser(description="Train multi-slot query encoder on OR pairs or mixtures.")
    p.add_argument("--mixtures", action="store_true", help="Use mixture data (queries with components, k_target, qrels)")
    p.add_argument("--data", type=str, default=None, help="Dataset dir (default: datasets/cars or datasets/cars_mixtures)")
    p.add_argument("--batch-size", type=int, default=16, help="Micro-batch size")
    p.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation (effective batch = batch_size * accum_steps)")
    p.add_argument("--epochs", type=int, default=30, help="Max epochs (10–30 typical; use early stopping or val to avoid overfit)")
    p.add_argument("--lr", type=float, default=5e-5, help="LR (2e-5 to 5e-5 stable for attention/proj heads)")
    p.add_argument("--amp", action="store_true", help="Use mixed precision (faster, less memory)")
    args = p.parse_args()

    data_dir = Path(args.data) if args.data else (MIXTURES_DIR if args.mixtures else DATA_DIR)
    batch_size = args.batch_size
    accum_steps = args.accum_steps
    epochs = args.epochs
    lr = args.lr
    use_amp = args.amp and torch.cuda.is_available()
    effective_batch = batch_size * accum_steps

    torch.manual_seed(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mixtures:
        print(f"Mixture training; data_dir={data_dir}")
    else:
        pairs = load_or_train_pairs(data_dir)
        print(f"Loaded {len(pairs)} OR training pairs from {data_dir}")
        if not pairs:
            raise SystemExit("No OR queries in train set. Run generate_cars_dataset.py first.")
        print(f"Config: batch_size={batch_size}, accum_steps={accum_steps} (effective={effective_batch}), epochs={epochs}, lr={lr}, amp={use_amp}")

    model = MultiSlotQwenQueryEncoder(k_max=K_MAX)
    for par in model.backbone.parameters():
        par.requires_grad = False
    model.backbone.eval()

    backbone_device = next(model.backbone.parameters()).device
    model.attn = model.attn.to(backbone_device)
    model.proj = model.proj.to(backbone_device)
    model.act = model.act.to(backbone_device)
    model.slots.data = model.slots.data.to(backbone_device)
    trainable = [model.slots, *list(model.attn.parameters()), *list(model.proj.parameters()), *list(model.act.parameters())]
    opt = torch.optim.AdamW(trainable, lr=lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if args.mixtures:
        train_mixtures(args, data_dir, model, backbone_device, opt, scaler, use_amp)
    else:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            opt.zero_grad()
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                composite = [b[0] for b in batch]
                sub1 = [b[1] for b in batch]
                sub2 = [b[2] for b in batch]

                with torch.no_grad():
                    t1 = model.get_single_vector_backbone(sub1, backbone_device)
                    t2 = model.get_single_vector_backbone(sub2, backbone_device)
                    t1 = t1.float().to(DEVICE)
                    t2 = t2.float().to(DEVICE)

                formatted = [format_query(q) for q in composite]
                enc = model.tokenizer(
                    formatted,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                enc = {k: v.to(backbone_device) for k, v in enc.items()}

                def forward_and_loss():
                    emb, p_out = model(enc["input_ids"], enc["attention_mask"])
                    emb = emb.to(DEVICE)
                    p_out = p_out.to(DEVICE)
                    v0, v1 = emb[:, 0, :], emb[:, 1, :]
                    loss_01_b = (1 - F.cosine_similarity(v0, t1, dim=-1)) + (1 - F.cosine_similarity(v1, t2, dim=-1))
                    loss_10_b = (1 - F.cosine_similarity(v0, t2, dim=-1)) + (1 - F.cosine_similarity(v1, t1, dim=-1))
                    loss_decomp = torch.minimum(loss_01_b, loss_10_b).mean()
                    loss_div = diversity_loss(emb, margin=DIVERSITY_MARGIN)
                    loss_act = (1 - p_out[:, 0]).mean() + (1 - p_out[:, 1]).mean()
                    loss_unmatched = p_out[:, 2:].mean()
                    return loss_decomp + DIVERSITY_WEIGHT * loss_div + ACT_WEIGHT * loss_act + 0.2 * loss_unmatched

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = forward_and_loss() / accum_steps
                    scaler.scale(loss).backward()
                else:
                    loss = forward_and_loss() / accum_steps
                    loss.backward()

                total_loss += loss.item() * accum_steps
                n_batches += 1

                if n_batches % accum_steps == 0:
                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad()

            if n_batches % accum_steps != 0:
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
            n_updates = (n_batches + accum_steps - 1) // accum_steps
            print(f"Epoch {epoch+1}/{epochs}  avg_loss={total_loss / n_batches:.4f}  updates={n_updates}")

    save_path = OUT_DIR / "multi_slot.pt"
    torch.save({
        "slots": model.slots,
        "attn": model.attn.state_dict(),
        "proj": model.proj.state_dict(),
        "act": model.act.state_dict(),
        "k_max": model.k_max,
        "out_dim": model.out_dim,
        "hidden_size": model.hidden_size,
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "trained_on_mixtures": args.mixtures,
    }, save_path)
    print(f"Saved to {save_path}")
    print("Use scripts/eval_multi_slot.py (or similar) to evaluate slot retrieval.")


if __name__ == "__main__":
    main()
