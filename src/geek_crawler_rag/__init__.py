"""Geek-Crawler-Rag — Python index + query API for the Geek-Crawler Mongo corpus."""

from geek_crawler_rag.app import app
from geek_crawler_rag.config import get_settings


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "geek_crawler_rag.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


__all__ = ["app", "main"]
