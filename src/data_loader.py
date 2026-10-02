from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader, UnstructuredPDFLoader,TextLoader,CSVLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader, UnstructuredEPubLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all documents from the specified directory and return a list of dictionaries containing
    the document content and metadata.

    Returns:
        List[Any]: A list of dictionaries containing the document content and metadata.
    """
    data_path = Path(data_dir).resolve()
    print(f"Loading documents from: {data_path}")
    documents = []

    # Load PDF files
    for pdf_file in data_path.glob("*.pdf"):
        loader = UnstructuredPDFLoader(str(pdf_file))
        docs = loader.load()
        documents.extend(docs)

    # Load text files
    for txt_file in data_path.glob("*.txt"):
        loader = TextLoader(str(txt_file))
        docs = loader.load()
        documents.extend(docs)

    # Load CSV files
    for csv_file in data_path.glob("*.csv"):
        loader = CSVLoader(str(csv_file))
        docs = loader.load()
        documents.extend(docs)

    # Load Word files
    for docx_file in data_path.glob("*.docx"):
        loader = DocxLoader(str(docx_file))
        docs = loader.load()
        documents.extend(docs)

    # Load PowerPoint files
    for pptx_file in data_path.glob("*.pptx"):
        loader = UnstructuredPowerPointLoader(str(pptx_file))
        docs = loader.load()
        documents.extend(docs)

    # Load Markdown files
    for md_file in data_path.glob("*.md"):
        loader = UnstructuredMarkdownLoader(str(md_file))
        docs = loader.load()
        documents.extend(docs)

    # Load HTML files
    for html_file in data_path.glob("*.html"):
        loader = UnstructuredHTMLLoader(str(html_file))
        docs = loader.load()
        documents.extend(docs)

    # Load EPub files
    for epub_file in data_path.glob("*.epub"):
        loader = UnstructuredEPubLoader(str(epub_file))
        docs = loader.load()
        documents.extend(docs)

    # Load Excel files
    for xlsx_file in data_path.glob("*.xlsx"):
        loader = UnstructuredExcelLoader(str(xlsx_file))
        docs = loader.load()
        documents.extend(docs)

    return documents