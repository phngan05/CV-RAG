"""
config.py
---------
Single source of truth for all application configuration.

Loads parameters from config.yaml and resolves secret values (API keys)
from environment variables (or a .env file). Every other module imports
`Settings` from here.

Usage:
    from config import Settings
    cfg = Settings()                        # auto-loads config.yaml + .env
    cfg = Settings(config_path="my.yaml")   # custom config file

    cfg.llm.model                           # "llama-3.3-70b-versatile"
    cfg.pinecone.index_name                 # "cv-rag-index"
    cfg.api.groq_api_key                  # resolved secret string
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# TYPED SUB-CONFIG DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class APIConfig:
    groq_api_key: str = ""
    pinecone_api_key: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "APIConfig":
        """Resolve env-var names → actual secret values."""
        return cls(
            groq_api_key=os.getenv(raw.get("groq_api_key_env", "GROQ_API_KEY"), ""),
            pinecone_api_key=os.getenv(raw.get("pinecone_api_key_env", "PINECONE_API_KEY"), ""),
        )

    def validate(self) -> None:
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Set them in your .env file or shell environment."
            )


@dataclass
class LLMConfig:
    model: str = "llama-3.3-70b-versatile"
    extraction_temperature: float = 0.0
    rag_temperature: float = 0.2
    outlier_temperature: float = 0.3
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, raw: dict) -> "LLMConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class EmbeddingsConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"
    normalize: bool = True
    dimension: int = 384

    @classmethod
    def from_dict(cls, raw: dict) -> "EmbeddingsConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class PineconeConfig:
    index_name: str = "cv-rag-index"
    cloud: str = "aws"
    region: str = "us-east-1"
    metric: str = "cosine"

    @classmethod
    def from_dict(cls, raw: dict) -> "PineconeConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class IngestionConfig:
    data_source: str = "snehaanbhawal/resume-dataset"  # Kaggle dataset identifier
    download_path: str = "./resumes"
    data_dir: str = "./resumes/data/data"
    chunk_size: int = 800
    chunk_overlap: int = 120
    ocr_text_threshold: int = 100
    ocr_dpi: int = 300
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    supported_extensions: List[str] = field(default_factory=lambda: [".pdf", ".PDF"])
    domain_folders: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "IngestionConfig":
        return cls(
            data_source=raw.get("data_source", "snehaanbhawal/resume-dataset"),
            download_path=raw.get("download_path", "./resumes"),
            data_dir=raw.get("data_dir", "./resumes/data/data"),
            chunk_size=raw.get("chunk_size", 800),
            chunk_overlap=raw.get("chunk_overlap", 120),
            ocr_text_threshold=raw.get("ocr_text_threshold", 100),
            ocr_dpi=raw.get("ocr_dpi", 300),
            ocr_languages=raw.get("ocr_languages", ["en"]),
            supported_extensions=raw.get("supported_extensions", [".pdf", ".PDF"]),
            domain_folders=raw.get("domain_folders", {}),
        )


@dataclass
class RetrievalConfig:
    default_k: int = 8
    outlier_k: int = 20
    similarity_k: int = 5
    search_type: str = "similarity"

    @classmethod
    def from_dict(cls, raw: dict) -> "RetrievalConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class EvaluationConfig:
    output_path: str = "./eval_results.json"
    metrics: List[str] = field(
        default_factory=lambda: [
            "context_precision", "context_recall",
            "faithfulness", "answer_correctness",
        ]
    )

    @classmethod
    def from_dict(cls, raw: dict) -> "EvaluationConfig":
        return cls(
            output_path=raw.get("output_path", "./eval_results.json"),
            metrics=raw.get("metrics", []),
        )


@dataclass
class AppConfig:
    page_title: str = "CV Intelligence Platform"
    page_icon: str = ":robot:"
    layout: str = "wide"
    max_upload_size_mb: int = 10
    chat_history_limit: int = 50

    @classmethod
    def from_dict(cls, raw: dict) -> "AppConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    @classmethod
    def from_dict(cls, raw: dict) -> "LoggingConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────
# ROOT SETTINGS CLASS
# ─────────────────────────────────────────────

class Settings:
    """
    Central configuration object.

    Loads config.yaml, resolves API keys from environment variables, and
    exposes typed sub-config objects.
    
    """

    # Default config file location
    DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        # 1. Load .env file first so os.getenv() finds the secrets
        env_path = env_file or Path(__file__).parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)

        # 2. Parse YAML
        cfg_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # 3. Build typed sub-configs
        self.api        = APIConfig.from_dict(raw.get("api", {}))
        self.llm        = LLMConfig.from_dict(raw.get("llm", {}))
        self.embeddings = EmbeddingsConfig.from_dict(raw.get("embeddings", {}))
        self.pinecone   = PineconeConfig.from_dict(raw.get("pinecone", {}))
        self.ingestion  = IngestionConfig.from_dict(raw.get("ingestion", {}))
        self.retrieval  = RetrievalConfig.from_dict(raw.get("retrieval", {}))
        self.evaluation = EvaluationConfig.from_dict(raw.get("evaluation", {}))
        self.app        = AppConfig.from_dict(raw.get("app", {}))
        self.logging    = LoggingConfig.from_dict(raw.get("logging", {}))

        # 4. Configure logging globally
        logging.basicConfig(level=self.logging.level, format=self.logging.format)

    def validate(self) -> "Settings":
        """Validate that required secrets are present. Call before using the app."""
        self.api.validate()
        return self

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"llm={self.llm.model}, "
            f"index={self.pinecone.index_name}, "
            f"embed={self.embeddings.model_name.split('/')[-1]}"
            f")"
        )


# ─────────────────────────────────────────────
# MODULE-LEVEL SINGLETON (optional convenience)
# ─────────────────────────────────────────────

_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Return the module-level Settings singleton.
    First call initialises it; subsequent calls return the cached instance.
    Pass config_path only on the first call (or to override).
    """
    global _settings
    if _settings is None or config_path is not None:
        _settings = Settings(config_path=config_path)
    return _settings
