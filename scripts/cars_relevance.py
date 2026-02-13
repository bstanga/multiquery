"""
Shared relevance logic for the cars dataset: load corpus, load base aspects (single-intent queries),
and compute Rel(q) = relevant doc ids for a given domain + attributes.
Used by build_mixtures.py and any code that needs aspect-level relevance.
"""
import json
from pathlib import Path
from typing import Any


def doc_matches(doc: dict, domain: str | None = None, **kwargs: Any) -> bool:
    """True if doc metadata matches domain and all given attribute filters."""
    m = doc.get("metadata") or {}
    if domain is not None and m.get("domain") != domain:
        return False
    if m.get("domain") == "car":
        if "color" in kwargs and m.get("color") != kwargs["color"]:
            return False
        if "brand" in kwargs and m.get("brand") != kwargs["brand"]:
            return False
        if "body_type" in kwargs and m.get("body_type") != kwargs["body_type"]:
            return False
        if "speed_tier" in kwargs and m.get("speed_tier") != kwargs["speed_tier"]:
            return False
        if "powertrain" in kwargs and m.get("powertrain") != kwargs["powertrain"]:
            return False
    elif m.get("domain") == "real_estate":
        if "rooms" in kwargs and m.get("rooms") != kwargs["rooms"]:
            return False
        if "property_type" in kwargs and m.get("property_type") != kwargs["property_type"]:
            return False
        if "price_tier" in kwargs and m.get("price_tier") != kwargs["price_tier"]:
            return False
    return True


def relevant_doc_ids(corpus: list[dict], domain: str, **attrs: Any) -> list[str]:
    """Return list of doc _id that match domain and attrs."""
    return [d["_id"] for d in corpus if doc_matches(d, domain=domain, **attrs)]


def load_corpus(data_dir: Path) -> list[dict]:
    """Load corpus.jsonl; each doc has _id, title?, text, metadata."""
    corpus = []
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus


def load_base_queries(data_dir: Path) -> list[dict]:
    """Load queries whose _id starts with 'single_' or 'single_multi_'. Each has _id, text, metadata (domain, attrs)."""
    base = []
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("_id", "")
            if qid.startswith("single_") or qid.startswith("single_multi_"):
                base.append(obj)
    return base


def aspects_with_relevance(corpus: list[dict], base_queries: list[dict]) -> list[tuple[str, str, set[str]]]:
    """
    For each base query, compute Rel(q) and return (qid, text, set(doc_ids)).
    Requires base_queries to have 'metadata' with 'domain' and 'attrs' (dict of attribute filters).
    """
    result = []
    for q in base_queries:
        meta = q.get("metadata") or {}
        domain = meta.get("domain")
        attrs = meta.get("attrs") or {}
        if not domain:
            continue
        rel = relevant_doc_ids(corpus, domain, **attrs)
        result.append((q["_id"], q["text"], set(rel)))
    return result
