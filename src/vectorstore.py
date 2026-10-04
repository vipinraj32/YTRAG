import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from src.embedding import EmbeddingPipeline
from sentence_transformers import SentenceTransformer

class FaisseVectorStore:
    def __init__(self, persist_directory: str="faiss_store", embedding_model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model_name
        self.model=SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.load()
        print(f"[INFO] Initialized FaisseVectorStore with embedding model: {embedding_model_name}")

    def build_from_documents(self,documents: List[Any]):
        print(f"[INFO] Building FAISS index from {len(documents)} documents...")
        self.embedding_pipeline = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = self.embedding_pipeline.chunk_documents(documents)
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype(np.float32), metadatas)
        self.save()
        print(f"[INFO] FAISS index built and saved to {self.persist_directory}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]=None):
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            print(f"[INFO] Created new FAISS index with dimension {dimension}")

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {len(embeddings)} embeddings to the FAISS index")

    def save(self):
        faiss_path = os.path.join(self.persist_directory, "faiss.index")
        metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved FAISS index to {faiss_path}")
        print(f"[INFO] Saved metadata to {metadata_path}")

    def load(self):
        faiss_path = os.path.join(self.persist_directory, "faiss.index")
        metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(faiss_path)
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded FAISS index from {faiss_path}")
            print(f"[INFO] Loaded metadata from {metadata_path}")
        else:
            print(f"[INFO] No existing FAISS index found at {self.persist_directory}. Starting fresh.")

    def search(self, query: np.ndarray, top_k: int = 5):
       D,I= self.index.search(query, top_k)
       results = []
       for idx, dist in zip(I[0], D[0]):
           metadata = self.metadata[idx] if idx < len(self.metadata) else None
           results.append({"index": idx, "metadata": metadata, "distance": dist})
       return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying FAISS index for: {query_text}")
        query_embedding = self.model.encode([query_text], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype(np.float32)
        return self.search(query_embedding, top_k=top_k)