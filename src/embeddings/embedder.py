# src/embeddings/embedder.py

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]):
        return self.model.encode(texts).tolist()