"""
ingest.py
---------
Modern PDF ingestion pipeline using SmartPDFLoader (Hybrid Digital/OCR).

Classes:
    DocumentDownloader - Handles downloading and local setup of the dataset.
    DocumentLoader     - Uses SmartPDFLoader (from pdf_loader.py) to handle both digital and scanned PDFs.
    DocumentIngester   - Splits Markdown by headers and upserts to Pinecone.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import Settings
from pdf_loader import SmartPDFLoader  # Integration with the new loader
import kagglehub

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CLASS: DocumentDownloader
# ─────────────────────────────────────────────

class DocumentDownloader:
    """Handles downloading and local setup of the dataset."""
    def __init__(self, settings: Settings):
        self.dataset_handle = settings.ingestion.data_source
        self.target_base_dir = Path(settings.ingestion.data_dir)
        self.data_root = Path(settings.ingestion.download_path)
        self.domains = settings.ingestion.domain_folders

    def download(self) -> str:
        if all((self.target_base_dir / d).exists() for d in self.domains):
            logger.info("Dataset already exists locally.")
            return str(self.target_base_dir)
        
        logger.info("Downloading dataset: %s", self.dataset_handle)
        download_path = kagglehub.dataset_download(self.dataset_handle)
        logger.info("Dataset downloaded to: %s", download_path)
        return download_path


# ─────────────────────────────────────────────
# CLASS: DocumentLoader
# ─────────────────────────────────────────────

class DocumentLoader:
    """
    Bridge between the ingestion pipeline and the SmartPDFLoader.
    Decides between digital extraction and OCR fallback automatically.
    """
    def __init__(self, settings: Settings):
        self.settings = settings

    def load(self, pdf_path: str, category: str) -> List[Document]:
        """
        Loads PDF using SmartPDFLoader and attaches domain category metadata.
        """
        # Using the factory method from pdf_loader.py
        loader = SmartPDFLoader.from_config(pdf_path, self.settings)
        
        # SmartPDFLoader returns a list of LangChain Document objects
        docs = loader.load()
        
        # Enrich metadata for multi-tenant / domain-filtered RAG.
        abs_path = str(Path(pdf_path).resolve())
        basename = Path(pdf_path).name
        for doc in docs:
            doc.metadata["category"]  = category
            doc.metadata["file_path"] = abs_path   # full path for PDF download
            doc.metadata["source"]    = basename   # filename for display & LLM tagging

        return docs


# ─────────────────────────────────────────────
# CLASS: DocumentIngester
# ─────────────────────────────────────────────

class DocumentIngester:
    """Orchestrates the full flow: Load -> Split -> Embed -> Upsert."""
    def __init__(self, settings: Settings):
        self._api_cfg = settings.api
        self._ing_cfg = settings.ingestion
        self._emb_cfg = settings.embeddings
        self._pc_cfg  = settings.pinecone

        self.loader = DocumentLoader(settings)
        self._embeddings = self._build_embeddings()

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._ing_cfg.chunk_size,
            chunk_overlap=self._ing_cfg.chunk_overlap,
        )

    def ingest(self):
        """Main entry point for the ingestion process."""
        data_path = Path(self._ing_cfg.data_dir)
        if not data_path.exists():
            logger.error("Data directory not found: %s", data_path)
            return

        all_docs = []
        for domain in self._ing_cfg.domain_folders:
            domain_path = data_path / domain
            if not domain_path.exists():
                continue

            pdf_files = list(domain_path.glob("*.pdf"))
            logger.info("Processing domain '%s' (%d files)", domain, len(pdf_files))

            for pdf_path in pdf_files:
                docs = self.loader.load(str(pdf_path), category=domain)
                all_docs.extend(docs)

        if not all_docs:
            logger.warning("No documents found to ingest.")
            return

        # Markdown Processing
        logger.info("Splitting %d documents into chunks...", len(all_docs))
        header_splits = []
        for doc in all_docs:
            # Preserve metadata when splitting
            splits = self._header_splitter.split_text(doc.page_content)
            for s in splits:
                s.metadata.update(doc.metadata)
            header_splits.extend(splits)

        final_chunks = self._text_splitter.split_documents(header_splits)
        logger.info("Final chunk count: %d", len(final_chunks))

        self._upsert(final_chunks)
        logger.info("=== Ingestion Complete: %d chunks in Pinecone ===", len(final_chunks))

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self._emb_cfg.model_name,
            model_kwargs={"device": self._emb_cfg.device},
            encode_kwargs={"normalize_embeddings": self._emb_cfg.normalize},
        )

    def _ensure_index_exists(self) -> None:
        pc = Pinecone(api_key=self._api_cfg.pinecone_api_key)
        if not pc.has_index(self._pc_cfg.index_name):
            logger.info("Creating index: %s", self._pc_cfg.index_name)
            pc.create_index(
                name=self._pc_cfg.index_name,
                dimension=self._emb_cfg.dimension,
                metric=self._pc_cfg.metric,
                spec=ServerlessSpec(cloud=self._pc_cfg.cloud, region=self._pc_cfg.region),
            )

    def _upsert(self, chunks: List[Document]):
        self._ensure_index_exists()
        logger.info("Upserting %d chunks to Pinecone...", len(chunks))
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            index_name=self._pc_cfg.index_name,
            pinecone_api_key=self._api_cfg.pinecone_api_key,
        )

# ─────────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from config import get_settings
    logging.basicConfig(level=logging.INFO)

    cfg = get_settings().validate()
    
    # 1. Setup Data
    downloader = DocumentDownloader(cfg)
    downloader.download()

    # 2. Run Ingestion
    ingester = DocumentIngester(cfg)
    ingester.ingest()