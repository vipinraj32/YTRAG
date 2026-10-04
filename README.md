# YTRAG

YTRAG is a lightweight Python Retrieval-Augmented Generation (RAG) project that turns local documents into searchable knowledge using FAISS and SentenceTransformers. It loads supported files from a data directory, chunks and embeds them, stores the vectors locally, and answers user questions using an Anthropic LLM.

## Overview

This project is designed for fast local experimentation with document-based Q&A. The workflow is:

1. Load documents from a local folder
2. Split them into chunks
3. Generate embeddings using a sentence-transformer model
4. Store embeddings in a FAISS index
5. Query the index for relevant context
6. Summarize the retrieved context with an LLM

## Features

- Multi-format document ingestion for PDFs, text files, CSV, Markdown, HTML, EPub, Excel, and other supported file types
- Sentence-transformer embeddings for semantic search
- FAISS vector store with persistence to disk
- Retrieval + summarization pipeline using Anthropic models
- Simple Python app entry point for quick testing

## Project Structure

```text
YTRAG/
├── app.py                 # Example script for running a sample query
├── data/                  # Source documents used by the RAG pipeline
├── faiss_store/           # Persisted FAISS index and metadata
├── src/
│   ├── data_loader.py     # Loads documents from the filesystem
│   ├── embedding.py       # Chunking and embedding pipeline
│   ├── search.py         # Search + LLM summarization
│   ├── vectorstore.py    # FAISS vector store implementation
│   └── ytrag/
│       └── __init__.py   # Package entry point
├── .env                   # Local environment variables (not committed)
├── pyproject.toml         # Project metadata and dependencies
├── requirement.txt        # Basic dependency list
├── README.md              # Project documentation
└── uv.lock                # Lock file used by uv
```

## Data and Indexing

Place your files inside the `data/` folder before building the vector store. The project can ingest supported document formats such as PDFs and text documents, and the system will persist the generated FAISS index in `faiss_store/`.

If the index does not exist yet, the application will build it automatically from the documents in `data/`.

## Example Workflow

```python
from src.data_loader import load_all_documents
from src.search import SearchEngine

docs = load_all_documents("data")
search_engine = SearchEngine(persist_directory="faiss_store")
answer = search_engine.search_and_summarize("What is in this document?", top_k=3)
print(answer)
```

## Notes

- This project is intended as a local RAG prototype and is easy to extend.
- The vector index is stored on disk, so repeated queries are faster once the index is built.
- The quality of the answer depends on the document quality, chunk size, and retrieval settings.

## License

This project does not currently include a license file. Add one if you intend to distribute or publish the project publicly.
