from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

if __name__ == "__main__":
    docs=load_all_documents("data")
    chuncks = EmbeddingPipeline().chunk_documents(docs)
    chunckVectors = EmbeddingPipeline().embed_chunks(chuncks)
    print(chunckVectors)
