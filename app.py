from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from vectorstore import FaisseVectorStore
from src.search import SearchEngine

if __name__ == "__main__":
    # docs=load_all_documents("data/pdf")
    # store=FaisseVectorStore("faiss_store")
    # store.load()
    rag_search=SearchEngine()
    query="what is the main purpose of the Military College of material management?"
    summary=rag_search.search_and_summarize(query, top_k=3)
    print(summary) 
   