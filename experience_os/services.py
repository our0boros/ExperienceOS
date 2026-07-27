"""统一模型服务层：Chat 和 Embedding 的唯一入口。

上层通过 :class:`Services` 获取注入的服务实例；provider 细节只在本模块实现。
通过 :class:`ProviderRegistry` 管理多个 LLM/Embedding provider 的注册与配置。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
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
    """统一 embedding 服务：多级后端自动降级 + SQLite 缓存，支持单条与批量计算。

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
            cached = self._storage.get_embedding(self._cache_text(text))
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_vecs = self._compute_batch(uncached_texts)
            for i, vec in zip(uncached_indices, new_vecs):
                results[i] = vec
                self._dimension = len(vec)
                self._storage.save_embedding(
                    self._cache_text(texts[i]), vec, self.model_name
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

    def _cache_text(self, text: str) -> str:
        return f"{self._config.backend or 'unknown'}/{self.model_name}\n{text}"

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


class ChatService:
    """OpenAI-compatible chat service."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key or "unused")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self._config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self.chat(messages, **kwargs)

    def complete_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        return self.chat_json(messages, **kwargs)

    def chat_json(self, messages: list[dict[str, str]], *, model: Optional[str] = None) -> dict:
        try:
            text = self.chat(messages, model=model, temperature=0.2,
                             response_format={"type": "json_object"})
        except Exception:
            text = self.chat(messages, model=model, temperature=0.2)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            block = text.split("```json", 1)[1].split("```", 1)[0] if "```json" in text else text.split("```", 1)[1].split("```", 1)[0]
            return json.loads(block)

    def tool_call(self, messages: list[dict], tools: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._config.model, messages=messages, tools=tools,
            tool_choice="auto", temperature=0.3,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            args = json.loads(call.function.arguments) if call.function.arguments else {}
            rendered = ", ".join(
                f'{key}="{value}"' if isinstance(value, str) else f"{key}={value}"
                for key, value in args.items()
            )
            return f"{call.function.name}({rendered})"
        return msg.content or ""

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._config.model, messages=messages, temperature=0.3, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def ping(self) -> bool:
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=5)
            return True
        except Exception as exc:
            log.error("LLM ping failed (%s): %s", self._config.backend, exc)
            return False

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

    chat: ChatService
    embedding: EmbeddingService
    config: Config

    def __init__(
        self,
        chat: ChatService,
        embedding: EmbeddingService,
        config: Config,
    ) -> None:
        self.chat = chat
        self.embedding = embedding
        self.config = config

    @classmethod
    def from_config(cls, config: Config, storage: Storage) -> "Services":
        """从 Config + Storage 创建完整服务栈。"""
        return cls(
            chat=ChatService(config.llm),
            embedding=EmbeddingService(config.llm, storage),
            config=config,
        )

    @classmethod
    def from_provider(cls, provider_name: str, config: Config,
                      storage: Storage) -> "Services":
        """从注册的 provider 名称创建服务栈。

        Args:
            provider_name: 注册的 provider 名称（如 "deepinfra"、"ollama"）。
            config: Config 对象（provider 信息会被写入 config.llm）。
            storage: Storage 实例。

        Returns:
            配置好的 Services 实例。
        """
        provider = ProviderRegistry.get(provider_name)
        if provider is not None:
            config.llm.backend = provider.name
            config.llm.base_url = provider.base_url
            if provider.llm_model:
                config.llm.model = provider.llm_model
            if provider.api_key_env:
                import os
                key = os.environ.get(provider.api_key_env, "")
                if key:
                    config.llm.api_key = key
            config.llm.embedding_model = (
                provider.embedding_model or config.llm.embedding_model
            )
        return cls.from_config(config, storage)

    @staticmethod
    def list_providers() -> list[str]:
        """列出所有已注册的 provider 名称。"""
        return ProviderRegistry.list_names()


# ──────────────────────────────────────────────────────────────────
# Provider registry
# ──────────────────────────────────────────────────────────────────


@dataclass
class ProviderInfo:
    """单个 LLM/Embedding provider 的注册信息。"""

    name: str                           # 短名称（如 "deepinfra"）
    base_url: str = ""                  # OpenAI-compatible API base URL
    api_key_env: str = ""               # API key 环境变量名
    llm_model: str = ""                 # 默认聊天模型
    embedding_model: str = ""           # 默认 embedding 模型
    embedding_dimension: int = 1024     # embedding 向量维度
    description: str = ""               # 人类可读描述

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "description": self.description,
        }


class ProviderRegistry:
    """LLM/Embedding provider 注册表。

    用法::

        # 注册新 provider
        ProviderRegistry.register(ProviderInfo(
            name="my_provider",
            base_url="https://api.example.com/v1",
            api_key_env="MY_API_KEY",
            llm_model="my-model-v1",
        ))

        # 按名称获取
        info = ProviderRegistry.get("deepinfra")

        # 列出全部
        for name in ProviderRegistry.list_names():
            print(name)
    """

    _providers: dict[str, ProviderInfo] = {}

    @classmethod
    def register(cls, info: ProviderInfo) -> None:
        """注册一个 provider（同名会覆盖）。"""
        cls._providers[info.name] = info

    @classmethod
    def get(cls, name: str) -> Optional[ProviderInfo]:
        """按名称获取 provider 信息。"""
        return cls._providers.get(name)

    @classmethod
    def list_names(cls) -> list[str]:
        """列出所有已注册的 provider 名称。"""
        return sorted(cls._providers.keys())

    @classmethod
    def list_all(cls) -> list[ProviderInfo]:
        """列出所有已注册的 provider。"""
        return sorted(cls._providers.values(), key=lambda p: p.name)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查 provider 是否已注册。"""
        return name in cls._providers

    @classmethod
    def resolve_url(cls, name: str) -> Optional[str]:
        """解析 provider 的 base URL。"""
        info = cls.get(name)
        return info.base_url if info else None


# ── 内置 provider 注册 ───────────────────────────────────────────

ProviderRegistry.register(ProviderInfo(
    name="deepinfra",
    base_url="https://api.deepinfra.com/v1/openai",
    api_key_env="DEEPINFRA_TOKEN",
    llm_model="deepseek-ai/DeepSeek-V4-Flash",
    embedding_model="Qwen/Qwen3-Embedding-8B",
    embedding_dimension=1024,
    description="DeepInfra — serverless LLM + embedding API",
))

ProviderRegistry.register(ProviderInfo(
    name="ollama",
    base_url="http://localhost:11434/v1",
    api_key_env="",
    llm_model="qwen2.5:7b",
    embedding_model="qwen2.5:7b",
    embedding_dimension=1024,
    description="Ollama — local LLM + embedding",
))

ProviderRegistry.register(ProviderInfo(
    name="openai",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    llm_model="gpt-4o",
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536,
    description="OpenAI — GPT + text-embedding",
))

ProviderRegistry.register(ProviderInfo(
    name="anthropic",
    base_url="",
    api_key_env="ANTHROPIC_API_KEY",
    llm_model="claude-sonnet-4-20250514",
    embedding_model="",
    embedding_dimension=0,
    description="Anthropic — Claude (no native embedding API; use external)",
))

ProviderRegistry.register(ProviderInfo(
    name="local",
    base_url="http://localhost:11434/v1",
    api_key_env="",
    llm_model="qwen2.5:7b",
    embedding_model="Qwen/Qwen3-Embedding-8B",
    embedding_dimension=1024,
    description="Local — Ollama LLM + sentence-transformers embedding (GPU if available)",
))

ProviderRegistry.register(ProviderInfo(
    name="litellm",
    base_url="http://localhost:4000/v1",
    api_key_env="LITELLM_API_KEY",
    llm_model="",
    embedding_model="",
    embedding_dimension=0,
    description="LiteLLM proxy — routes to multiple providers",
))
