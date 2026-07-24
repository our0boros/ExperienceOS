"""Configuration for ExperienceOS.

All settings are read from environment variables with sensible defaults so the
framework runs out-of-the-box with a local ``ollama`` instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class LLMConfig:
    """OpenAI-compatible LLM backend configuration."""

    backend: str = field(default_factory=lambda: _env("EOS_LLM_BACKEND", "ollama"))

    # ollama
    ollama_base_url: str = field(
        default_factory=lambda: _env("EOS_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )
    ollama_model: str = field(default_factory=lambda: _env("EOS_OLLAMA_MODEL", "qwen2.5:7b"))
    ollama_api_key: str = field(default_factory=lambda: _env("EOS_OLLAMA_API_KEY", "ollama"))

    # DeepInfra
    deepinfra_base_url: str = field(
        default_factory=lambda: _env("EOS_DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
    )
    deepinfra_model: str = field(
        default_factory=lambda: _env("EOS_DEEPINFRA_MODEL", "MiniMaxAI/MiniMax-M2.7")
    )
    deepinfra_api_key: str = field(
        default_factory=lambda: _env("DEEPINFRA_TOKEN", "")
    )

    # embeddings
    embedding_model: str = field(
        default_factory=lambda: _env("EOS_EMBEDDING_MODEL", "qwen2.5:7b")
    )
    embedding_dim: int = field(default_factory=lambda: int(_env("EOS_EMBEDDING_DIM", "3584")))

    # ---- active values resolved at runtime --------------------------------
    @property
    def base_url(self) -> str:
        if self.backend == "deepinfra":
            return self.deepinfra_base_url
        return self.ollama_base_url

    @property
    def model(self) -> str:
        if self.backend == "deepinfra":
            return self.deepinfra_model
        return self.ollama_model

    @property
    def api_key(self) -> str:
        if self.backend == "deepinfra":
            return self.deepinfra_api_key
        return self.ollama_api_key

    @property
    def embed_base_url(self) -> str:
        """Embeddings always go through ollama (local, free)."""
        return self.ollama_base_url

    @property
    def embed_api_key(self) -> str:
        return self.ollama_api_key

    def __post_init__(self) -> None:
        if self.backend == "deepinfra" and not self.deepinfra_api_key:
            raise ValueError(
                "DEEPINFRA_TOKEN is not set. Either set it or use EOS_LLM_BACKEND=ollama."
            )


@dataclass
class InductionConfig:
    """Thresholds governing *when* to compile a harness."""

    min_support: int = field(default_factory=lambda: int(_env("EOS_MIN_SUPPORT", "3")))
    validation_threshold: float = field(
        default_factory=lambda: float(_env("EOS_VALIDATION_THRESHOLD", "0.8"))
    )
    f2_patch_trigger: int = 2  # consecutive F2 failures before patching
    max_replay_steps: int = 50


@dataclass
class Config:
    """Top-level configuration object."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    induction: InductionConfig = field(default_factory=InductionConfig)
    data_dir: Path = field(
        default_factory=lambda: Path(_env("EOS_DATA_DIR", ".experience_os_data"))
    )

    def ensure_dirs(self) -> None:
        """Create the on-disk repository directory tree."""
        for sub in ("trajectories", "records", "harnesses", "embeddings", "stats"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)
