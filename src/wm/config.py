"""Runtime configuration, read from the environment (and a local ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
DEFAULT_DB_PATH = "rag_chroma_db"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ConfigError(RuntimeError):
    """A required setting is missing."""


@dataclass(frozen=True)
class Settings:
    model: str
    effort: str
    db_path: Path
    embedding_model: str
    anthropic_api_key: str | None
    football_api_key: str | None

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            model=os.getenv("WM_MODEL", DEFAULT_MODEL),
            effort=os.getenv("WM_EFFORT", DEFAULT_EFFORT),
            db_path=Path(os.getenv("WM_DB_PATH", DEFAULT_DB_PATH)),
            embedding_model=os.getenv("WM_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            football_api_key=(os.getenv("FOOTBALL_API_KEY") or "").strip() or None,
        )

    def require_football_key(self) -> str:
        if not self.football_api_key:
            raise ConfigError("FOOTBALL_API_KEY is not set. Add your API-Football key to .env.")
        return self.football_api_key
