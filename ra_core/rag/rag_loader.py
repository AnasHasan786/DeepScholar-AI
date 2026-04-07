from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import *
from typing import Optional, List, Tuple
import tempfile
import os


class RAGLoader:
    def __init__(
        self,
        llm_model,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        k=RETRIEVER_K,
    ):
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.retriever = None

    def load_multiple_pdfs(self, files: List[Tuple[bytes, str]]):
        """
        Processes a list of (file_bytes, filename) tuples and builds a
        unified vector store for comparison, including author metadata extraction.
        """
        all_chunks = []
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        for file_bytes, filename in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                # --- Metadata Enrichment ---
                # Attempt to extract Author from PDF properties
                pdf_metadata = docs[0].metadata if docs else {}
                # PyPDFLoader usually stores the original PDF info in 'author' or 'Author'
                extracted_author = pdf_metadata.get(
                    "author", pdf_metadata.get("Author", "Unknown Author")
                )

                # Add source details to metadata for each document/page
                for doc in docs:
                    doc.metadata["source_file"] = filename
                    doc.metadata["authors"] = extracted_author

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", " ", ""],
                )
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if not all_chunks:
            raise ValueError("No text could be extracted from the uploaded PDFs.")

        # Build one unified vector store containing all papers
        vector_store = FAISS.from_documents(all_chunks, embeddings)

        self.retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return self.retriever

    def load_pdf(self, file_bytes: bytes, filename: str = "document.pdf"):
        """
        Legacy support for single PDF loading.
        """
        return self.load_multiple_pdfs([(file_bytes, filename)])
