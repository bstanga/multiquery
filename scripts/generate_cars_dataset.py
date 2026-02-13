#!/usr/bin/env python3
"""
Generate a toy BEIR-format dataset of cars + real estate with attributes.
Supports single-intent queries (e.g. "fast car", "3.5 room apartment") and
OR queries including cross-domain (e.g. "fast car or 3.5 room apartment").
"""

import json
import random
from pathlib import Path

# Default sizes
NUM_CARS = 800
NUM_REAL_ESTATE = 800
NUM_SINGLE_QUERIES = 100
NUM_OR_QUERIES = 200
NUM_CROSS_DOMAIN_OR = 80  # OR queries that mix car + real estate
TEST_RATIO = 0.3

# Car attributes
COLORS = ["red", "blue", "white", "black", "silver", "green"]
BRANDS = ["Toyota", "Honda", "Ford", "Tesla", "BMW", "Chevrolet", "Nissan"]
BODY_TYPES = ["sedan", "SUV", "sports car", "truck", "hatchback"]
SPEED_TIERS = [
    ("slow", 70, 95),
    ("moderate", 96, 125),
    ("fast", 126, 180),
]
POWERTRAINS = ["gas", "electric", "hybrid"]

# Real estate attributes
ROOMS = [2, 2.5, 3, 3.5, 4, 4.5, 5]
PROPERTY_TYPES = ["apartment", "house", "condo"]
PRICE_TIERS = [
    ("cheap", 150, 280),
    ("moderate", 281, 450),
    ("expensive", 451, 800),
]


def main():
    random.seed(42)
    out_dir = Path(__file__).resolve().parent.parent / "datasets" / "cars"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qrels").mkdir(exist_ok=True)

    corpus = []

    # ---- 1a) Cars ----
    for i in range(NUM_CARS):
        color = random.choice(COLORS)
        brand = random.choice(BRANDS)
        body = random.choice(BODY_TYPES)
        speed_name, low, high = random.choice(SPEED_TIERS)
        top_speed = random.randint(low, high)
        powertrain = random.choice(POWERTRAINS)
        doc_id = f"car_{i}"
        text = (
            f"A {color} {brand} {body}. "
            f"{powertrain.capitalize()} powertrain. "
            f"Top speed {top_speed} mph."
        )
        corpus.append({
            "_id": doc_id,
            "title": f"{brand} {body}",
            "text": text,
            "metadata": {
                "domain": "car",
                "color": color,
                "brand": brand,
                "body_type": body,
                "speed_tier": speed_name,
                "top_speed": top_speed,
                "powertrain": powertrain,
            },
        })

    # ---- 1b) Real estate ----
    for i in range(NUM_REAL_ESTATE):
        rooms = random.choice(ROOMS)
        ptype = random.choice(PROPERTY_TYPES)
        price_name, low_k, high_k = random.choice(PRICE_TIERS)
        price_k = random.randint(low_k, high_k)
        doc_id = f"real_{i}"
        text = (
            f"{rooms} room {ptype} for sale. "
            f"{price_name.capitalize()} price range, {price_k}k."
        )
        corpus.append({
            "_id": doc_id,
            "title": f"{rooms} room {ptype}",
            "text": text,
            "metadata": {
                "domain": "real_estate",
                "rooms": rooms,
                "property_type": ptype,
                "price_tier": price_name,
                "price_k": price_k,
            },
        })

    with open(out_dir / "corpus.jsonl", "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    doc_by_id = {d["_id"]: d for d in corpus}

    def doc_matches(doc: dict, domain: str = None, **kwargs) -> bool:
        m = doc["metadata"]
        if domain is not None and m.get("domain") != domain:
            return False
        if m.get("domain") == "car":
            if "color" in kwargs and m["color"] != kwargs["color"]:
                return False
            if "brand" in kwargs and m["brand"] != kwargs["brand"]:
                return False
            if "body_type" in kwargs and m["body_type"] != kwargs["body_type"]:
                return False
            if "speed_tier" in kwargs and m["speed_tier"] != kwargs["speed_tier"]:
                return False
            if "powertrain" in kwargs and m["powertrain"] != kwargs["powertrain"]:
                return False
        elif m.get("domain") == "real_estate":
            if "rooms" in kwargs and m["rooms"] != kwargs["rooms"]:
                return False
            if "property_type" in kwargs and m["property_type"] != kwargs["property_type"]:
                return False
            if "price_tier" in kwargs and m["price_tier"] != kwargs["price_tier"]:
                return False
        return True

    def relevant_doc_ids(domain: str, **kwargs) -> list[str]:
        return [d["_id"] for d in corpus if doc_matches(d, domain=domain, **kwargs)]

    # ---- 2) Single-intent templates (cars + real estate) ----
    single_templates = []
    for color in COLORS:
        single_templates.append((f"{color} car", "car", {"color": color}))
    for body in BODY_TYPES:
        single_templates.append((body, "car", {"body_type": body}))
    for speed_name, _, _ in SPEED_TIERS:
        single_templates.append((f"{speed_name} car", "car", {"speed_tier": speed_name}))
    for pt in POWERTRAINS:
        single_templates.append((f"{pt} car", "car", {"powertrain": pt}))
    for brand in BRANDS:
        single_templates.append((f"{brand} car", "car", {"brand": brand}))
    single_templates.append(("fast vehicle", "car", {"speed_tier": "fast"}))
    single_templates.append(("electric vehicle", "car", {"powertrain": "electric"}))

    for rooms in ROOMS:
        single_templates.append((f"{rooms} room apartment", "real_estate", {"rooms": rooms, "property_type": "apartment"}))
        single_templates.append((f"{rooms} room house", "real_estate", {"rooms": rooms, "property_type": "house"}))
    for ptype in PROPERTY_TYPES:
        single_templates.append((f"{ptype} for sale", "real_estate", {"property_type": ptype}))
    for price_name, _, _ in PRICE_TIERS:
        single_templates.append((f"{price_name} apartment", "real_estate", {"price_tier": price_name, "property_type": "apartment"}))
    single_templates.append(("3.5 room apartment", "real_estate", {"rooms": 3.5, "property_type": "apartment"}))
    single_templates.append(("4 room house", "real_estate", {"rooms": 4, "property_type": "house"}))

    random.shuffle(single_templates)
    single_templates = single_templates[:NUM_SINGLE_QUERIES]

    single_queries = []
    for i, (text, domain, attrs) in enumerate(single_templates):
        qid = f"single_{i}"
        rel = relevant_doc_ids(domain, **attrs)
        single_queries.append({
            "_id": qid,
            "text": text,
            "metadata": {"domain": domain, "attrs": attrs},
        })
        single_queries[-1]["_relevant"] = rel

    # ---- 2b) Multi-attribute single-intent (harder: intersection of attrs) ----
    multi_templates = [
        ("fast red car", "car", {"speed_tier": "fast", "color": "red"}),
        ("cheap 3.5 room apartment", "real_estate", {"rooms": 3.5, "property_type": "apartment", "price_tier": "cheap"}),
        ("expensive 4 room house", "real_estate", {"rooms": 4, "property_type": "house", "price_tier": "expensive"}),
        ("electric SUV", "car", {"powertrain": "electric", "body_type": "SUV"}),
        ("moderate 3 room apartment", "real_estate", {"rooms": 3, "property_type": "apartment", "price_tier": "moderate"}),
        ("blue sports car", "car", {"color": "blue", "body_type": "sports car"}),
        ("cheap 2 room apartment", "real_estate", {"rooms": 2, "property_type": "apartment", "price_tier": "cheap"}),
        ("black fast car", "car", {"color": "black", "speed_tier": "fast"}),
        ("expensive 5 room house", "real_estate", {"rooms": 5, "property_type": "house", "price_tier": "expensive"}),
        ("hybrid sedan", "car", {"powertrain": "hybrid", "body_type": "sedan"}),
        ("cheap condo", "real_estate", {"property_type": "condo", "price_tier": "cheap"}),
        ("white truck", "car", {"color": "white", "body_type": "truck"}),
        ("expensive 3.5 room apartment", "real_estate", {"rooms": 3.5, "property_type": "apartment", "price_tier": "expensive"}),
        ("slow red car", "car", {"speed_tier": "slow", "color": "red"}),
        ("moderate 4.5 room house", "real_estate", {"rooms": 4.5, "property_type": "house", "price_tier": "moderate"}),
    ]
    multi_queries = []
    for i, (text, domain, attrs) in enumerate(multi_templates):
        qid = f"single_multi_{i}"
        rel = relevant_doc_ids(domain, **attrs)
        if len(rel) < 2:
            continue
        multi_queries.append({
            "_id": qid,
            "text": text,
            "metadata": {"domain": domain, "attrs": attrs, "multi_attr": True},
        })
        multi_queries[-1]["_relevant"] = rel

    # ---- 3) OR queries: same-domain + cross-domain ----
    car_templates = [(t, d, a) for t, d, a in single_templates if d == "car"]
    re_templates = [(t, d, a) for t, d, a in single_templates if d == "real_estate"]
    or_queries = []
    used_pairs = set()

    def add_or(t1, t2):
        key = (t1[0], t2[0])
        if key in used_pairs:
            return False
        used_pairs.add(key)
        text = f"{t1[0]} or {t2[0]}"
        rel1 = set(relevant_doc_ids(t1[1], **t1[2]))
        rel2 = set(relevant_doc_ids(t2[1], **t2[2]))
        rel_union = rel1 | rel2
        if len(rel_union) < 2:
            return False
        or_queries.append({
            "_id": f"or_{len(or_queries)}",
            "text": text,
            "metadata": {"left": t1[2], "right": t2[2], "left_domain": t1[1], "right_domain": t2[1]},
        })
        or_queries[-1]["_relevant"] = list(rel_union)
        return True

    # Cross-domain OR (car or real estate) - e.g. "fast car or 3.5 room apartment"
    attempts = 0
    while len(or_queries) < NUM_CROSS_DOMAIN_OR and attempts < 400:
        attempts += 1
        t1 = random.choice(car_templates)
        t2 = random.choice(re_templates)
        add_or(t1, t2)

    # Same-domain OR (car or car, real_estate or real_estate)
    attempts = 0
    while len(or_queries) < NUM_OR_QUERIES and attempts < 500:
        attempts += 1
        t1 = random.choice(single_templates)
        t2 = random.choice(single_templates)
        if t1[1] != t2[1]:
            continue
        if t1[2] == t2[2]:
            continue
        add_or(t1, t2)

    # ---- 4) Write queries.jsonl ----
    all_queries = single_queries + multi_queries + or_queries
    with open(out_dir / "queries.jsonl", "w") as f:
        for q in all_queries:
            out = {"_id": q["_id"], "text": q["text"]}
            if "metadata" in q:
                out["metadata"] = q["metadata"]
            f.write(json.dumps(out) + "\n")

    # ---- 5) Qrels: train and test split ----
    # First NUM_CROSS_DOMAIN_OR are cross-domain, rest same-domain
    or_cross = or_queries[:NUM_CROSS_DOMAIN_OR]
    or_same = or_queries[NUM_CROSS_DOMAIN_OR:]
    n_test_single = max(1, int(len(single_queries) * TEST_RATIO))
    n_test_multi = max(0, int(len(multi_queries) * TEST_RATIO))
    n_test_or_cross = max(1, int(len(or_cross) * TEST_RATIO))
    n_test_or_same = max(1, int(len(or_same) * TEST_RATIO))
    random.shuffle(single_queries)
    random.shuffle(multi_queries)
    random.shuffle(or_cross)
    random.shuffle(or_same)
    test_queries = (
        single_queries[:n_test_single]
        + multi_queries[:n_test_multi]
        + or_cross[:n_test_or_cross]
        + or_same[:n_test_or_same]
    )
    train_queries = (
        single_queries[n_test_single:]
        + multi_queries[n_test_multi:]
        + or_cross[n_test_or_cross:]
        + or_same[n_test_or_same:]
    )

    def write_qrels(queries_with_rel: list, path: Path):
        with open(path, "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for q in queries_with_rel:
                for doc_id in q["_relevant"]:
                    if doc_id in doc_by_id:
                        f.write(f"{q['_id']}\t{doc_id}\t1\n")

    write_qrels(test_queries, out_dir / "qrels" / "test.tsv")
    write_qrels(train_queries, out_dir / "qrels" / "train.tsv")

    # ---- 6) Hard test set: cross-domain OR + multi-attr single (for --test --hard) ----
    hard_queries = or_cross + multi_queries
    write_qrels(hard_queries, out_dir / "qrels" / "test_hard.tsv")

    n_cross = len(or_cross)
    n_same = len(or_same)
    print(f"Wrote {out_dir}")
    print(f"  corpus:       {NUM_CARS} cars + {NUM_REAL_ESTATE} real estate = {len(corpus)} docs")
    print(f"  queries:      {len(single_queries)} single, {len(multi_queries)} multi-attr, {n_cross} cross-domain OR, {n_same} same-domain OR")
    print(f"  qrels/test:   {len(test_queries)} queries")
    print(f"  qrels/train:  {len(train_queries)} queries")
    print(f"  qrels/test_hard: {len(hard_queries)} queries (cross-domain OR + multi-attr single)")


if __name__ == "__main__":
    main()
