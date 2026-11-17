"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# OpenAI-compatible endpoints. Groq, Gemini and a self-hosted vLLM server all
# speak the same protocol, so one client covers every provider.
PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    # Project 3 (StoryServe): your own vLLM endpoint.
    "selfhosted": {
        "base_url": os.getenv("SELFHOSTED_BASE_URL", "http://localhost:8000/v1"),
        "key_env": "SELFHOSTED_API_KEY",
        "default_model": "meta-llama/Llama-3.1-8B-Instruct",
    },
    # Offline deterministic fake, used by tests and `--mock` demo runs.
    "mock": {"base_url": "", "key_env": "", "default_model": "mock"},
}

# USD per 1M tokens (input, output). Free tiers cost $0, but tracking the
# list price tells you what the pipeline would cost at scale. Edit freely.
PRICING: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "mock": (0.0, 0.0),
}


def _env(name: str, default: str, cast=str):
    """Read the environment when Settings() is created, not at import time."""
    return field(default_factory=lambda: cast(os.getenv(name, default)))


def _bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _list(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class Settings:
    provider: str = _env("LLM_PROVIDER", "groq")
    # A different provider for the critic avoids a model grading its own work
    # (self-preference bias). Falls back to the main provider.
    critic_provider: str = field(
        default_factory=lambda: os.getenv("CRITIC_PROVIDER") or os.getenv("LLM_PROVIDER", "groq")
    )
    temperature: float = _env("LLM_TEMPERATURE", "0.7", float)
    max_retries: int = _env("LLM_MAX_RETRIES", "5", int)
    max_repair_attempts: int = _env("MAX_REPAIR_ATTEMPTS", "2", int)

    num_episodes: int = _env("NUM_EPISODES", "5", int)
    target_languages: list[str] = _env("TARGET_LANGUAGES", "hi", _list)
    critic_threshold: float = _env("CRITIC_THRESHOLD", "3.8", float)
    max_revisions: int = _env("MAX_REVISIONS", "2", int)
    max_tool_rounds: int = _env("MAX_TOOL_ROUNDS", "4", int)
    use_rag: bool = _env("USE_RAG", "true", _bool)

    embed_model: str = _env("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    dedup_threshold: float = _env("DEDUP_THRESHOLD", "0.90", float)
    qdrant_url: str = _env("QDRANT_URL", "")  # empty = in-memory Qdrant
    qdrant_api_key: str = _env("QDRANT_API_KEY", "")

    output_dir: str = _env("OUTPUT_DIR", "outputs")

    def model(self, stage: str, provider: str | None = None) -> str:
        """Model for a stage: <STAGE>_MODEL, else LLM_MODEL (main provider only), else provider default."""
        provider = provider or self.provider
        override = os.getenv(f"{stage.upper()}_MODEL")
        if override:
            return override
        if provider == self.provider and os.getenv("LLM_MODEL"):
            return os.getenv("LLM_MODEL", "")
        return PROVIDERS[provider]["default_model"]


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, v)
    return s
