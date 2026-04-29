from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, documents):
        pass

    @abstractmethod
    def search(self, query_vector, filters=None, top_k=5):
        pass