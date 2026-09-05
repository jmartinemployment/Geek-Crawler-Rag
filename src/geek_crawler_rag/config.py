from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mongo_crawler_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "geek_crawler"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "geek_crawler_chunks"

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Mid of plan range 500–800 tokens; small overlap.
    chunk_size_tokens: int = 650
    chunk_overlap_tokens: int = 80

    page_batch_size: int = 25
    embed_batch_size: int = 64

    # Optional shared secret for index/query (GeekAPI / gcc-v2). Empty = open (dev).
    api_key: str | None = None

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
