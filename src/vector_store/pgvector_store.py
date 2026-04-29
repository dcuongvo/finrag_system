from .base import VectorStore


class PgVectorStore(VectorStore):
    def __init__(self):
        raise NotImplementedError(
            "PgVectorStore will be implemented later for PostgreSQL / AWS RDS."
        )

    def upsert(self, documents):
        pass

    def search(self, query_vector, filters=None, top_k=5):
        pass