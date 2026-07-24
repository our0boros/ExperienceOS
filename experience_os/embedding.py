"""本地 Embedding 客户端。

优先使用本地 Qwen3-Embedding-8B 模型（通过 sentence-transformers），
回退到 ollama embeddings API，最后回退到 hash 伪向量。

支持 SQLite 向量持久化缓存（通过 Storage 层）。

使用方式::

    from experience_os.embedding import EmbeddingClient
    from experience_os.config import Config
    from experience_os.storage import Storage

    config = Config()
    storage = Storage(config)
    embedder = EmbeddingClient(config, storage)

    vec = embedder.embed("find user by email")
    vec = embedder.embed_batch(["text1", "text2"])
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Optional

from experience_os.config import Config

log = logging.getLogger(__name__)


class EmbeddingClient:
    """多级回退的 embedding 客户端。

    优先级：
      1. 本地 Qwen3-Embedding-8B（sentence-transformers，GPU 加速）
      2. ollama embeddings API
      3. hash 伪向量（仅保证一致性，无语义）

    所有 embedding 通过 Storage 层持久化到 SQLite，避免重复计算。
    """

    def __init__(
        self,
        config: Config,
        storage: Optional[Any] = None,
        model_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self._model = None
        self._model_loaded = False
        self._dim: int | None = None
        self._model_name = ""

        # 确定 embedding 模型路径
        if model_path:
            self._model_path = model_path
        else:
            # 检查 models/ 目录下的 embedding 模型
            project_root = config.data_dir.parent
            candidates = [
                project_root / "models" / "Qwen3-Embedding-8B",
            ]
            self._model_path = None
            for p in candidates:
                if p.exists():
                    self._model_path = str(p)
                    break

    def _load_model(self) -> bool:
        """加载本地 sentence-transformers 模型。"""
        if self._model_loaded:
            return self._model is not None

        self._model_loaded = True
        if not self._model_path:
            log.debug("No local embedding model path configured")
            return False

        try:
            from sentence_transformers import SentenceTransformer

            device = "cuda" if _cuda_available() else "cpu"
            log.info("Loading embedding model from %s (device=%s)", self._model_path, device)
            self._model = SentenceTransformer(self._model_path, device=device)
            self._dim = self._model.get_sentence_embedding_dimension()
            self._model_name = Path(self._model_path).name
            log.info(
                "Embedding model loaded: %s, dim=%d", self._model_name, self._dim
            )
            return True
        except ImportError:
            log.warning("sentence-transformers not installed, falling back to ollama")
            return False
        except Exception as exc:
            log.warning("Failed to load embedding model %s: %s", self._model_path, exc)
            return False

    def embed(self, text: str) -> list[float]:
        """生成单条文本的 embedding 向量。

        先查 SQLite 缓存，命中则直接返回；
        未命中则计算并持久化。
        """
        # 1. 查缓存
        if self.storage:
            cached = self.storage.get_embedding(text)
            if cached is not None:
                return cached

        # 2. 计算新向量
        vec = self._compute_embedding(text)

        # 3. 持久化
        if self.storage:
            self.storage.save_embedding(text, vec, self._model_name or "ollama")

        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding。"""
        if not texts:
            return []

        # 检查缓存
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        for i, text in enumerate(texts):
            if self.storage:
                cached = self.storage.get_embedding(text)
                if cached is not None:
                    results[i] = cached
                    continue
            uncached_indices.append(i)

        # 计算未缓存的
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            new_vecs = self._compute_batch(uncached_texts)
            for idx, vec in zip(uncached_indices, new_vecs):
                results[idx] = vec
                if self.storage:
                    self.storage.save_embedding(
                    texts[idx], vec, self._model_name or "ollama"
                    )

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # 实际计算逻辑
    # ------------------------------------------------------------------
    def _compute_embedding(self, text: str) -> list[float]:
        return self._compute_batch([text])[0]

    def _compute_batch(self, texts: list[str]) -> list[list[float]]:
        """按优先级计算 embedding。"""
        # 1. 本地模型
        if self._load_model():
            try:
                vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return [v.tolist() for v in vecs]
            except Exception as exc:
                log.warning("Local model encode failed: %s, falling back", exc)

        # 2. ollama API
        try:
            return self._ollama_embed_batch(texts)
        except Exception as exc:
            log.warning("Ollama embeddings failed: %s, using hash fallback", exc)

        # 3. hash fallback
        dim = self._dim or self.config.llm.embedding_dim
        return [_hash_embedding(t, dim) for t in texts]

    def _ollama_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """通过 ollama OpenAI 兼容 API 获取 embedding。"""
        from openai import OpenAI

        client = OpenAI(
            base_url=self.config.llm.embed_base_url,
            api_key=self.config.llm.embed_api_key or "unused",
        )
        resp = client.embeddings.create(
            model=self.config.llm.embedding_model,
            input=texts,
        )
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    @property
    def dimension(self) -> int:
        """当前 embedding 维度。"""
        if self._dim is not None:
            return self._dim
        return self.config.llm.embedding_dim

    @property
    def model_name(self) -> str:
        return self._model_name or f"ollama/{self.config.llm.embedding_model}"


def _cuda_available() -> bool:
    """检查 CUDA 是否可用。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _hash_embedding(text: str, dim: int) -> list[float]:
    """确定性 hash 伪向量（无语义，仅保证一致性）。"""
    vec = [0.0] * dim
    data = text.encode()
    for i in range(0, max(len(data), dim * 4), dim):
        chunk = data[i : i + dim] if i < len(data) else b""
        h = hashlib.sha256((str(i) + ":").encode() + chunk).digest()
        for j in range(0, len(h), 4):
            idx = (i + j) % dim
            val = int.from_bytes(h[j : j + 4], "little") / 0xFFFFFFFF
            vec[idx] += val
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
