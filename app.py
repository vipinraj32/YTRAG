from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from vectorstore import FaisseVectorStore

if __name__ == "__main__":
    docs=load_all_documents("data/pdf")
    store = FaisseVectorStore("faiss_store")
    store.build_from_documents(docs)
    
