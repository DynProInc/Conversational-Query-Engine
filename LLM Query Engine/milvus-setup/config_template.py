# Database Schema RAG Configuration
# Copy this file to config.py and update with your settings

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class MilvusConfig:
    """Milvus database configuration"""
    host: str = "localhost"
    port: str = "19530"
    user: str = ""
    password: str = ""
    secure: bool = False

    # Collection settings
    collection_name: str = "database_schema_collection"
    index_type: str = "IVF_FLAT"
    metric_type: str = "IP"  # Inner Product
    nlist: int = 128
    nprobe: int = 10

@dataclass 
class EmbeddingConfig:
    """Embedding model configuration"""
    provider: str = "openai"  # "openai" or "sentence_transformers"

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-small"
    openai_dimensions: int = 1536

    # Sentence Transformers settings
    sentence_model: str = "all-MiniLM-L6-v2"
    sentence_dimensions: int = 384

@dataclass
class RAGConfig:
    """RAG pipeline configuration"""
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.7

    # Generation settings
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1500

    # Token optimization
    enable_token_counting: bool = True
    max_context_tokens: int = 4000

@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Performance
    batch_size: int = 100
    max_workers: int = 4

    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds

# Load configuration from environment variables
def get_config():
    """Load configuration with environment variable overrides"""

    milvus_config = MilvusConfig(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
        user=os.getenv("MILVUS_USER", ""),
        password=os.getenv("MILVUS_PASSWORD", ""),
    )

    embedding_config = EmbeddingConfig(
        provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "text-embedding-3-small"),
    )

    rag_config = RAGConfig(
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7")),
        llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
    )

    system_config = SystemConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
    )

    return milvus_config, embedding_config, rag_config, system_config
