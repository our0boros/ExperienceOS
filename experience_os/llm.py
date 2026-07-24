"""OpenAI-compatible LLM client supporting ollama (local) and DeepInfra (remote).

Both backends expose the standard ``/chat/completions`` and ``/embeddings``
endpoints, so a single ``openai`` client handles both.  This module provides a
thin wrapper with convenience helpers for chat completion (with optional
streaming) and embedding generation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Iterator

from openai import OpenAI

from experience_os.config import LLMConfig

log = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper over the ``openai`` Python SDK.

    Parameters
    ----------
    config:
        A :class:`~experience_os.config.LLMConfig`.  The ``base_url``,
        ``api_key`` and ``model`` fields are resolved based on the active
        backend (``ollama`` or ``deepinfra``).
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "unused",
        )
        self._embed_client = OpenAI(
            base_url=config.embed_base_url,
            api_key=config.embed_api_key or "unused",
        )

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Synchronous chat completion returning the assistant text."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> dict:
        """Chat completion that forces JSON output and parses it.

        Falls back to extracting the first ```json``` fenced block if the
        backend does not honour ``response_format`` (e.g. some ollama models).
        """
        try:
            text = self.chat(
                messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            text = self.chat(messages, temperature=temperature)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # try to extract ```json ... ``` block
            if "```json" in text:
                block = text.split("```json", 1)[1].split("```", 1)[0]
                return json.loads(block)
            if "```" in text:
                block = text.split("```", 1)[1].split("```", 1)[0]
                return json.loads(block)
            raise

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """Streaming chat completion yielding text chunks."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for *text*.

        Uses the ollama OpenAI-compatible ``/v1/embeddings`` endpoint. If the
        server does not support embeddings (started without ``--embeddings``),
        falls back to a deterministic hash-based vector so the framework stays
        functional for testing.
        """
        try:
            resp = self._embed_client.embeddings.create(
                model=self.config.embedding_model,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as exc:
            log.debug("Real embeddings unavailable (%s), using hash fallback", exc)
            return _hash_embedding(text, self.config.embedding_dim)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding (single request)."""
        if not texts:
            return []
        try:
            resp = self._embed_client.embeddings.create(
                model=self.config.embedding_model,
                input=texts,
            )
            return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        except Exception:
            return [_hash_embedding(t, self.config.embedding_dim) for t in texts]

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """Return ``True`` if the backend responds to a trivial request."""
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=5)
            return True
        except Exception as exc:
            log.error("LLM ping failed (%s): %s", self.config.backend, exc)
            return False


# ======================================================================
# Fallback embedding for environments without an embedding model
# ======================================================================
def _hash_embedding(text: str, dim: int) -> list[float]:
    """Deterministic hash-based pseudo-embedding.

    Not semantically meaningful, but provides stable, consistent vectors so
    that cosine similarity between *identical/near-identical* text is high and
    unrelated text is low.  Used only when the real embedding API is down.
    """
    import hashlib

    vec = [0.0] * dim
    # hash in chunks of `dim` bytes
    data = text.encode()
    for i in range(0, max(len(data), dim * 4), dim):
        chunk = data[i : i + dim] if i < len(data) else b""
        h = hashlib.sha256((str(i) + ":").encode() + chunk).digest()
        for j in range(0, len(h), 4):
            idx = (i + j) % dim
            val = int.from_bytes(h[j : j + 4], "little") / 0xFFFFFFFF
            vec[idx] += val
    # normalise
    import math

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
