import argparse
import json
import numpy as np
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from sentence_transformers import SentenceTransformer

K_VALUES = [1, 3, 5, 10, 100]

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"

def format_query(q: str) -> str:
    # Recommended Qwen query template
    return f"Instruct: {INSTRUCT}\nQuery: {q}"

class STWrapper:
    """BEIR expects an object with encode_corpus / encode_queries."""
    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize = normalize

    def encode_queries(self, queries: list[str], batch_size: int = None, **kwargs):
        texts = [format_query(q) for q in queries]
        return self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def encode_corpus(self, corpus: list[dict], batch_size: int = None, **kwargs):
        # BEIR corpus entries are dicts with keys like title/text
        texts = []
        for doc in corpus:
            title = doc.get("title", "")
            body = doc.get("text", "")
            if title:
                texts.append(f"{title}\n{body}".strip())
            else:
                texts.append(body.strip())

        return self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

def load_query_k_target(data_path: Path) -> dict:
    """Load query_id -> k_target from mixture queries.jsonl (mix_* lines only)."""
    out = {}
    path = data_path / "queries.jsonl"
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("_id", "")
            if qid.startswith("mix_"):
                out[qid] = obj.get("k_target", 1)
    return out


def run_test(data_path: Path, use_hard: bool = False, by_k: bool = False):
    """Run evaluation on test set (or hard subset); print nDCG@10 and Recall@100."""
    if use_hard:
        hard_qrels_path = data_path / "qrels" / "test_hard.tsv"
        if not hard_qrels_path.exists():
            raise FileNotFoundError(f"Hard test set not found: {hard_qrels_path}. Run scripts/generate_cars_dataset.py first.")
        loader = GenericDataLoader(
            data_folder=str(data_path),
            qrels_file=str(hard_qrels_path),
        )
        corpus, queries, qrels = loader.load_custom()
        print("Evaluating on hard set (cross-domain OR + multi-attr single)...")
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split="test")
    model = STWrapper(MODEL_NAME, batch_size=64, normalize=True)
    dres = DRES(model, batch_size=64)
    retriever = EvaluateRetrieval(dres, score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)
    if by_k:
        query_to_k = load_query_k_target(data_path)
        if query_to_k:
            for k_val in [1, 2, 3]:
                qids_k = [qid for qid in queries if query_to_k.get(qid) == k_val]
                if not qids_k:
                    continue
                qrels_k = {qid: qrels[qid] for qid in qids_k if qid in qrels}
                results_k = {qid: results[qid] for qid in qids_k if qid in results}
                if not qrels_k:
                    continue
                ndcg_k, _map_k, recall_k, _ = retriever.evaluate(qrels_k, results_k, K_VALUES)
                print(f"  K={k_val} (n={len(qids_k)}): nDCG@10={ndcg_k['NDCG@10']:.4f}, Recall@100={recall_k['Recall@100']:.4f}")
        else:
            print("--by-k: no mix_* queries with k_target in queries.jsonl; skipping breakdown.")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, K_VALUES)
    print("nDCG@10:", ndcg["NDCG@10"])
    print("Recall@100:", recall["Recall@100"])


def run_query(data_path: Path, query: str, top_k: int = 10):
    """Run a single query and print top-k results. Uses direct encode + cos_sim to avoid BEIR bug with 1 query."""
    corpus, _, _ = GenericDataLoader(data_folder=str(data_path)).load(split="test")
    model = STWrapper(MODEL_NAME, batch_size=64, normalize=True)
    corpus_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id] for doc_id in corpus_ids]
    query_emb = model.encode_queries([query], show_progress_bar=False)
    corpus_emb = model.encode_corpus(corpus_list, show_progress_bar=True)
    # Normalized embeddings -> cosine sim = dot product
    scores = np.dot(query_emb, corpus_emb.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    print(f"Query: {query}\nTop-{top_k} results:\n")
    for rank, idx in enumerate(top_indices, 1):
        doc_id = corpus_ids[idx]
        doc = corpus[doc_id]
        text = doc.get("text", "")
        title = doc.get("title", "")
        if title:
            text = f"{title} — {text}"
        print(f"  {rank}. [{doc_id}] (score={scores[idx]:.4f}) {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval: --test for eval, --query for single query.")
    parser.add_argument("--test", action="store_true", help="Run evaluation on test set")
    parser.add_argument("--hard", action="store_true", help="Use hard test set (cross-domain OR + multi-attr single); use with --test")
    parser.add_argument("--query", type=str, default=None, metavar="Q", help='Run a single query and print top results (e.g. --query "fast car or 3.5 room apartment")')
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to show for --query (default: 10)")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset folder (default: datasets/cars). Use datasets/cars_mixtures for mixture test set.")
    parser.add_argument("--by-k", action="store_true", help="Report metrics by k_target (K=1,2,3). Use with --test and mixture data.")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else Path(__file__).resolve().parent / "datasets" / "cars"

    if args.query is not None:
        run_query(data_path, args.query, top_k=args.top_k)
    if args.test:
        run_test(data_path, use_hard=args.hard, by_k=args.by_k)
    if args.query is None and not args.test:
        parser.print_help()
        print("\nUse --test to run evaluation or --query \"...\" to run a single query.")
