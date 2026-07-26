"""统一服务层：LLM / Embedding 的单一入口。

所有模块通过 :class:`Services` 依赖注入获取 LLM 和 embedding 能力，
不再各自创建客户端实例。后续 cost tracking、retry、circuit breaker
等横切关注点统一在此层实现。

使用方式::

    from experience_os.config import Config
    from experience_os.storage import Storage
    from experience_os.services import Services

    config = Config()
    storage = Storage(config)
    svc = Services.from_config(config, storage)

    # LLM 调用
    reply = svc.llm.chat([{"role": "user", "content": "hello"}])
    data = svc.llm.chat_json([{"role": "user", "content": "{...}"}])

    # Embedding + 意图匹配
    vec = svc.embed.embed("find user by email")
    matches = svc.embed.match_intent("find user by email", patterns)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Iterator, Optional

from openai import OpenAI

from experience_os.config import Config, LLMConfig
from experience_os.storage import Storage

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Embedding service
# ──────────────────────────────────────────────────────────────────


class EmbeddingService:
    """统一 embedding 服务：多级后端自动降级 + SQLite 缓存。

    后端优先级：
      1. 本地 Qwen3-Embedding-8B（sentence-transformers，GPU 加速）
      2. DeepInfra API ``Qwen/Qwen3-Embedding-8B``（远程）
      3. ollama embeddings API
      4. Hash 伪向量（确定性，无语义但保证可用）
    """

    def __init__(self, config: LLMConfig, storage: Storage) -> None:
        self._config = config
        self._storage = storage
        self._local_model = None  # lazy-loaded sentence-transformers
        self._remote_client: Optional[OpenAI] = None
        self._model_name: Optional[str] = None
        self._dimension: int = 1024  # Qwen3-Embedding-8B default

    # ── public API ──────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """单条 embedding，自动走缓存。"""
        text_hash = self._hash(text)
        cached = self._storage.get_embedding(text_hash)
        if cached is not None:
            return cached
        vec = self._compute(text)
        self._storage.save_embedding(text_hash, vec, self.model_name)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding：缓存命中的跳过，只计算未命中部分。"""
        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._storage.get_embedding(self._hash(text))
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_vecs = self._compute_batch(uncached_texts)
            for i, vec in zip(uncached_indices, new_vecs):
                results[i] = vec
                self._storage.save_embedding(
                    self._hash(texts[i]), vec, self.model_name
                )

        return results  # type: ignore[return-value]

    def match_intent(
        self,
        query: str,
        candidates: list,
        *,
        high_threshold: float = 0.85,
        low_threshold: float = 0.65,
    ) -> tuple[list[tuple], list[tuple], list]:
        """四层降级意图匹配。

        返回三元组 ``(high_conf, fuzzy, rejected)``：
        - **high_conf** (cosine ≥ high_threshold)：直接使用，不消耗 LLM
        - **fuzzy** (low_threshold ≤ cosine < high_threshold)：需 LLM 评估
        - **rejected** (cosine < low_threshold)：直接回退 ReAct，不浪费 LLM 成本
        """
        from experience_os.models import SubStepPattern

        query_vec = self.embed(query)
        high: list[tuple[SubStepPattern, float]] = []
        fuzzy: list[tuple[SubStepPattern, float]] = []
        rejected: list[SubStepPattern] = []

        for pattern in candidates:
            if not isinstance(pattern, SubStepPattern):
                raise TypeError(f"Expected SubStepPattern, got {type(pattern)}")
            if pattern.intent_embedding is None:
                pattern.intent_embedding = self.embed(pattern.description or pattern.intent)
            sim = self._cosine(query_vec, pattern.intent_embedding)
            if sim >= high_threshold:
                high.append((pattern, sim))
            elif sim >= low_threshold:
                fuzzy.append((pattern, sim))
            else:
                rejected.append(pattern)

        high.sort(key=lambda x: x[1], reverse=True)
        fuzzy.sort(key=lambda x: x[1], reverse=True)
        return high, fuzzy, rejected

    # ── properties ──────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        if self._model_name:
            return self._model_name
        if self._local_model is not None:
            self._model_name = "Qwen3-Embedding-8B-local"
        elif self._config.backend == "deepinfra":
            self._model_name = "Qwen/Qwen3-Embedding-8B"
        else:
            self._model_name = self._config.embedding_model or "qwen2.5:7b"
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    # ── internal: compute ───────────────────────────────────────

    def _compute(self, text: str) -> list[float]:
        """单条 embedding 的实际计算：逐级降级。"""
        # Level 1: local sentence-transformers
        try:
            model = self._get_local_model()
            if model is not None:
                vec = model.encode(text, normalize_embeddings=True)
                return vec.tolist()
        except Exception:
            log.debug("Local embedding failed, trying next backend")

        # Level 2: remote API (DeepInfra)
        try:
            client = self._get_remote_client()
            if client is not None:
                resp = client.embeddings.create(
                    model="Qwen/Qwen3-Embedding-8B",
                    input=text,
                    encoding_format="float",
                )
                vec = resp.data[0].embedding
                return vec
        except Exception:
            log.debug("Remote embedding failed, trying next backend")

        # Level 3: ollama
        try:
            vec = self._embed_via_ollama(text)
            if vec:
                return vec
        except Exception:
            log.debug("Ollama embedding failed, falling back to hash")

        # Level 4: hash pseudo-vector
        return self._hash_vector(text)

    def _compute_batch(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding 计算。"""
        # Try local model first (best for batch)
        try:
            model = self._get_local_model()
            if model is not None:
                vecs = model.encode(texts, normalize_embeddings=True)
                return [v.tolist() for v in vecs]
        except Exception:
            pass

        # Fall back to one-by-one
        return [self._compute(t) for t in texts]

    def _get_local_model(self):
        """Lazy-load sentence-transformers 模型。"""
        if self._local_model is not None:
            return self._local_model
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.environ.get("EOS_LOCAL_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
            # Try to find local model in shared cache
            local_path = Path("models/Qwen3-Embedding-8B")
            if local_path.exists():
                model_name = str(local_path.resolve())
            self._local_model = SentenceTransformer(model_name)
            log.info("Embedding: local Qwen3-Embedding-8B loaded")
            return self._local_model
        except ImportError:
            log.debug("sentence-transformers not installed, skipping local embedding")
            return None
        except Exception:
            log.debug("Failed to load local embedding model", exc_info=True)
            return None

    def _get_remote_client(self) -> Optional[OpenAI]:
        """Lazy-init DeepInfra API client for embeddings."""
        if self._remote_client is not None:
            return self._remote_client
        api_key = os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("DEEPINFRA_TOKEN") or ""
        if not api_key:
            return None
        self._remote_client = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=api_key,
        )
        return self._remote_client

    def _embed_via_ollama(self, text: str) -> Optional[list[float]]:
        """通过 ollama embeddings API 计算向量。"""
        try:
            model = self._config.embedding_model or "qwen2.5:7b"
            base_url = self._config.ollama_base_url or "http://localhost:11434/v1"
            client = OpenAI(base_url=base_url, api_key="ollama")
            resp = client.embeddings.create(model=model, input=text)
            return resp.data[0].embedding
        except Exception:
            return None

    def _hash_vector(self, text: str) -> list[float]:
        """确定性 SHA-256 伪向量（无语义，但稳定可用）。"""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        dim = self._dimension
        vec = []
        for i in range(dim):
            byte_val = digest[i % len(digest)]
            offset = (i // len(digest)) * 0.01
            vec.append(((byte_val / 255.0) * 2 - 1) + offset)
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    # ── utils ───────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0


# ──────────────────────────────────────────────────────────────────
# LLM service
# ──────────────────────────────────────────────────────────────────


class LLMService:
    """统一的 LLM 调用层：chat / chat_json / stream，后端切换对调用方透明。

    包装 ``LLMClient`` 并留出 cost tracking / retry / circuit breaker 扩展点。
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        # 委托到现有 LLMClient（保持兼容）
        from experience_os.llm import LLMClient
        self._client = LLMClient(config)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict] = None,
        model: Optional[str] = None,
    ) -> str:
        """单轮对话，返回文本。"""
        return self._client.chat(messages, response_format=response_format)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
    ) -> dict:
        """JSON 模式对话，返回 dict（含 markdown fence 提取回退）。"""
        return self._client.chat_json(messages)

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """流式对话。"""
        return self._client.stream(messages)

    def ping(self) -> bool:
        """测试 LLM 后端连通性。"""
        return self._client.ping()

    @property
    def model(self) -> str:
        return self._config.model


# ──────────────────────────────────────────────────────────────────
# Services container
# ──────────────────────────────────────────────────────────────────


class Services:
    """全局服务容器：一处初始化、处处复用。

    所有模块通过依赖注入接收 ``Services`` 实例，不再各自创建
    LLM / embedding 客户端。

    用法::

        svc = Services.from_config(config, storage)
        inductor = HarnessInductor(svc, repo)
    """

    llm: LLMService
    embed: EmbeddingService
    config: Config

    def __init__(
        self,
        llm: LLMService,
        embed: EmbeddingService,
        config: Config,
    ) -> None:
        self.llm = llm
        self.embed = embed
        self.config = config

    @classmethod
    def from_config(cls, config: Config, storage: Storage) -> "Services":
        """从 Config + Storage 创建完整服务栈。"""
        return cls(
            llm=LLMService(config.llm),
            embed=EmbeddingService(config.llm, storage),
            config=config,
        )
