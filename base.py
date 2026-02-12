import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from sentence_transformers import SentenceTransformer

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

# 1) Load BEIR dataset
dataset = "nfcorpus"  # or "nfcorpus"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# 2) Build retriever
model = STWrapper(MODEL_NAME, batch_size=64, normalize=True)
dres = DRES(model, batch_size=64)

retriever = EvaluateRetrieval(dres, score_function="dot") 
results = retriever.retrieve(corpus, queries)

# 3) Evaluate
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
print("nDCG@10:", ndcg["NDCG@10"])
print("Recall@100:", recall["Recall@100"])
