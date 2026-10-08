import os
from dotenv import load_dotenv
from src.vectorstore import FaisseVectorStore
from langchain_anthropic import ChatAnthropic

load_dotenv()

class SearchEngine:
    def __init__(self, persist_directory: str = "faiss_store", embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.vector_store = FaisseVectorStore(persist_directory=persist_directory, embedding_model_name=embedding_model_name)
        faiss_path = os.path.join(persist_directory, "faiss.index")
        metadata_path = os.path.join(persist_directory, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(metadata_path)):
            from data_loader import load_all_documents
            docs=load_all_documents("data")
            self.vector_store.build_from_documents(docs)
        else:
            self.vector_store.load()
            ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  
            self.llm = ChatAnthropic(model="claude-fable-5-1")
            print(f"[INFO] Loaded Anthropic LLM with API key: {ANTHROPIC_API_KEY}")

    def search_and_summarize(self, query_text: str, top_k: int = 3) -> str:
        results = self.vector_store.query(query_text, top_k=top_k)
        texts=[r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"Summarize the following context in relation to the query '{query_text}':\n\n{context}"
        response = self.llm.invoke(prompt)
        return response.content
