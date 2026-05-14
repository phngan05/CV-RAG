"""
pdf_loader.py
-------------
Unified PDF Loader for RAG pipeline.

Features:
- Auto-detect digital vs scanned PDF
- Fallback to OCR if needed
- Return LangChain Document objects
- Enrich metadata for downstream RAG + extraction
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from easyocr import Reader
from pdf2image import convert_from_path
from config import Settings
import numpy as np

logger = logging.getLogger(__name__)


# =========================================================
# OCR LOADER (for scanned PDFs)
# =========================================================

class OCRLoader(BaseLoader):
    _reader = None
    def __init__(self, file_path: str, dpi: int = 300, languages: List[str] = None):
        self.file_path = file_path
        self.dpi = dpi
        if OCRLoader._reader is None:
            OCRLoader._reader = Reader(languages, gpu=False, verbose=False)

    def load(self) -> List[Document]:
        documents = []
        try:
            images = convert_from_path(self.file_path, dpi=self.dpi)
            for i, page_image in enumerate(images):
                page_array = np.array(page_image)
                
                result = OCRLoader._reader.readtext(page_array, detail=0)
                page_text = "\n".join(result)
                
                doc = Document(
                    page_content=page_text,
                    metadata={
                    "source": self.file_path,
                    "page": i,
                    "loader": "pymupdf"}
                )
                documents.append(doc)
            
            
            if not documents:
                logger.warning(f"OCR run successfully but no character was extracted: {self.file_path}")

        except Exception as e:
            logger.error(f"OCR failed for {self.file_path}: {e}")
            

        return documents


# =========================================================
# SMART LOADER (main entry)
# =========================================================

class SmartPDFLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        ocr_threshold: int = 150,
        dpi: int = 300,
        languages: List[str] = ["en"]
    ):
        self.file_path = file_path
        self.ocr_threshold = ocr_threshold
        self.dpi = dpi
        self.languages = languages

    @classmethod
    def from_config(cls, file_path: str, settings: Settings):
        return cls(
            file_path=file_path,
            ocr_threshold=settings.ingestion.ocr_text_threshold,
            dpi=settings.ingestion.ocr_dpi,
            languages=settings.ingestion.ocr_languages
        )

    # =====================================================
    # MAIN LOAD FUNCTION
    # =====================================================

    def load(self) -> List[Document]:
        path = Path(self.file_path)

        if not path.exists():
            logger.error(f"File not found: {self.file_path}")
            return []

        # --- Try digital extraction first ---
        digital_docs = self._load_digital()

        total_chars = sum(len(doc.page_content) for doc in digital_docs)

        if total_chars >= self.ocr_threshold:
            logger.info(f"[DIGITAL] {path.name} ({total_chars} chars)")
            return digital_docs

        # --- Fallback to OCR ---
        logger.info(f"[OCR] {path.name} (only {total_chars} chars, using OCR)")
        return self._load_ocr()

    # =====================================================
    # DIGITAL LOADER
    # =====================================================

    def _load_digital(self) -> List[Document]:
        try:
            loader = PyMuPDF4LLMLoader(self.file_path)
            docs = loader.load()

            for i, doc in enumerate(docs, start=1):
                doc.metadata.update({
                    "source": self.file_path,
                    "page": i,
                    "loader": "pymupdf"
                })

            return docs

        except Exception as e:
            logger.error(f"Digital load failed: {e}")
            return []

    # =====================================================
    # OCR LOADER
    # =====================================================

    def _load_ocr(self) -> List[Document]:
        loader = OCRLoader(
            file_path=self.file_path,
            dpi=self.dpi,
            languages=self.languages
        )
        return loader.load()