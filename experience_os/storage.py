"""SQLite driver for the Store layer.

.. deprecated::
    本模块仅供 ``repository.py`` 和 ``experience_library.py`` 内部使用。
    新代码不应直接调用 ``Storage``；请使用 ``stores.py`` 的 Store facade。

本模块负责 SQLite 持久化、向量缓存和历史 JSON 数据迁移，不承担 domain facade 语义。

提供：
  - 结构化查询（按 task_type、outcome、时间范围等）
  - 向量持久化（embedding 存为 BLOB）
  - 环境 metadata 存储
  - 事务一致性

存储结构::

    experience_os.db
    ├── trajectories    -- Layer 0: 原始轨迹
    ├── records         -- Layer 1: 经验记录
    ├── harnesses       -- Layer 2: 可执行 Harness
    ├── stats           -- Layer 3: 任务类型统计
    ├── embeddings      -- 向量索引（text → vector BLOB）
    ├── env_metadata    -- 环境 metadata 快照
    └── schema_version  -- schema 版本管理

JSON 迁移仅用于导入历史数据；SQLite 是唯一运行时存储驱动。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Optional

from experience_os.config import Config

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
-- Layer 0: 轨迹
CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    task_description TEXT,
    outcome TEXT,
    phase TEXT DEFAULT '',
    steps_json TEXT,
    cot_json TEXT,
    env_snapshot_json TEXT,
    tokens_used INTEGER DEFAULT 0,
    latency_seconds REAL DEFAULT 0.0,
    created_at REAL
);

-- Layer 1: 经验记录
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    preconditions_json TEXT,
    canonical_steps_json TEXT,
    invariants_json TEXT,
    terminal_verifier TEXT,
    source_trajectories_json TEXT,
    created_at REAL
);

-- Layer 2: Harness
CREATE TABLE IF NOT EXISTS harnesses (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    capability TEXT,
    version INTEGER,
    parent_id TEXT,
    status TEXT,
    preconditions_json TEXT,
    invariants_json TEXT,
    params_json TEXT,
    procedure_code TEXT,
    verification_json TEXT,
    failure_counts_json TEXT,
    embedding BLOB,
    created_at REAL,
    updated_at REAL,
    branch TEXT DEFAULT 'main'
);

-- Layer 3: 统计
CREATE TABLE IF NOT EXISTS stats (
    task_type TEXT PRIMARY KEY,
    stats_json TEXT,
    updated_at REAL
);

-- 向量索引
CREATE TABLE IF NOT EXISTS embeddings (
    text_hash TEXT PRIMARY KEY,
    source_text TEXT,
    embedding BLOB,
    dim INTEGER,
    model_name TEXT,
    created_at REAL
);

-- 环境 metadata（结构化列，非 JSON blob）
CREATE TABLE IF NOT EXISTS env_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    -- OS
    os_system TEXT,
    os_release TEXT,
    os_distro TEXT,
    os_machine TEXT,
    -- Python
    python_version TEXT,
    python_executable TEXT,
    in_venv INTEGER,
    -- Hardware
    cpu_count INTEGER,
    memory_gb REAL,
    gpu_name TEXT,
    gpu_memory TEXT,
    -- Models
    ollama_models TEXT,
    local_model_names TEXT,
    -- τ-bench
    tau2_installed INTEGER,
    tau2_version TEXT,
    tau2_domains TEXT,
    -- Packages (variable dict, keep as JSON)
    packages_json TEXT,
    -- Env vars
    eos_llm_backend TEXT,
    eos_ollama_model TEXT,
    eos_deepinfra_model TEXT
);

-- Schema 版本
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_traj_type ON trajectories(task_type);
CREATE INDEX IF NOT EXISTS idx_traj_outcome ON trajectories(outcome);
CREATE INDEX IF NOT EXISTS idx_traj_type_outcome ON trajectories(task_type, outcome);
CREATE INDEX IF NOT EXISTS idx_harness_type ON harnesses(task_type);
CREATE INDEX IF NOT EXISTS idx_harness_status ON harnesses(status);
CREATE INDEX IF NOT EXISTS idx_record_type ON records(task_type);
"""


def _pack_vector(vec: list[float]) -> bytes:
    """将 float 列表打包为 bytes（float32 little-endian）。"""
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_vector(data: bytes) -> list[float]:
    """将 bytes 解包为 float 列表。"""
    if not data:
        return []
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


class Storage:
    """SQLite 存储层，替代纯 JSON 文件。

    所有数据存储在单个 ``experience_os.db`` 文件中，
    支持结构化查询和向量持久化。
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.db_path = config.data_dir / "experience_os.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库 schema。"""
        conn = self._conn or sqlite3.connect(self.db_path)
        conn.executescript(_SCHEMA_SQL)
        # 增量迁移：为旧库添加缺失列（如 phase）
        self._migrate_columns(conn)
        # 记录 schema 版本
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, time.time()),
        )
        conn.commit()
        self._conn = conn
        log.info("SQLite storage initialized at %s", self.db_path)

    @staticmethod
    def _migrate_columns(conn: sqlite3.Connection) -> None:
        """增量添加缺失列（兼容旧库）。"""
        # 检查 trajectories 表是否有 phase 列
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trajectories)").fetchall()}
        if "phase" not in cols:
            try:
                conn.execute("ALTER TABLE trajectories ADD COLUMN phase TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # 列已存在

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    # ==================================================================
    # 轨迹存储
    # ==================================================================
    def save_trajectory(self, traj: Any) -> None:
        """存储轨迹（dataclass 或 dict）。"""
        from experience_os.repository import _as_dict

        d = _as_dict(traj) if hasattr(traj, "__dict__") else traj
        steps = json.dumps(d.get("steps", []), ensure_ascii=False)
        cot = json.dumps(d.get("structured_cot", {}), ensure_ascii=False)
        env = json.dumps(d.get("env_snapshot", {}), ensure_ascii=False)
        self.conn.execute(
            """INSERT OR REPLACE INTO trajectories
            (id, task_id, task_type, task_description, outcome, phase, steps_json,
             cot_json, env_snapshot_json, tokens_used, latency_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                d["id"], d.get("task_id", ""), d.get("task_type", ""),
                d.get("task_description", ""), d.get("outcome", ""),
                d.get("phase", ""),
                steps, cot, env,
                d.get("tokens_used", 0), d.get("latency_seconds", 0.0),
                d.get("created_at", time.time()),
            ),
        )
        self.conn.commit()

    def load_trajectories(
        self, task_type: str = "", success_only: bool = False
    ) -> list[dict]:
        """查询轨迹。"""
        sql = "SELECT * FROM trajectories"
        clauses: list[str] = []
        params: list = []
        if task_type:
            clauses.append("task_type = ?")
            params.append(task_type)
        if success_only:
            clauses.append("outcome = 'success'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at"
        rows = self.conn.execute(sql, params).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM trajectories LIMIT 0").description]
        return [dict(zip(cols, row)) for row in rows]

    # ==================================================================
    # Harness 存储
    # ==================================================================
    def save_harness(self, harness: Any, embedding: list[float] | None = None) -> None:
        """存储 Harness 及其 embedding 向量。"""
        from experience_os.repository import _as_dict

        d = _as_dict(harness) if hasattr(harness, "__dict__") else harness
        embed_blob = _pack_vector(embedding) if embedding else None
        self.conn.execute(
            """INSERT OR REPLACE INTO harnesses
            (id, task_type, capability, version, parent_id, status,
             preconditions_json, invariants_json, params_json, procedure_code,
             verification_json, failure_counts_json, embedding,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                d["id"], d.get("task_type", ""), d.get("capability", ""),
                d.get("version", 1), d.get("parent_id"), d.get("status", "active"),
                json.dumps(d.get("preconditions", {}), ensure_ascii=False),
                json.dumps(d.get("invariants", []), ensure_ascii=False),
                json.dumps(d.get("params", []), ensure_ascii=False),
                d.get("procedure_code", ""),
                json.dumps(d.get("verification", {}), ensure_ascii=False),
                json.dumps(d.get("failure_counts", {}), ensure_ascii=False),
                embed_blob,
                d.get("created_at", time.time()),
                d.get("updated_at", time.time()),
            ),
        )
        self.conn.commit()

    def load_harnesses(self, task_type: str = "", active_only: bool = False) -> list[dict]:
        """查询 Harness。"""
        sql = "SELECT * FROM harnesses"
        clauses: list[str] = []
        params: list = []
        if task_type:
            clauses.append("task_type = ?")
            params.append(task_type)
        if active_only:
            clauses.append("status = 'active'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        rows = self.conn.execute(sql, params).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM harnesses LIMIT 0").description]
        return [dict(zip(cols, row)) for row in rows]

    def get_harness_embedding(self, harness_id: str) -> list[float] | None:
        """获取 Harness 的持久化 embedding。"""
        row = self.conn.execute(
            "SELECT embedding FROM harnesses WHERE id = ?", (harness_id,)
        ).fetchone()
        if row and row[0]:
            return _unpack_vector(row[0])
        return None

    # ==================================================================
    # 经验记录存储
    # ==================================================================
    def save_record(self, record: Any) -> None:
        """存储 ExperienceRecord（dataclass 或 dict）。"""
        from experience_os.repository import _as_dict

        d = _as_dict(record) if hasattr(record, "__dict__") else record
        self.conn.execute(
            """INSERT OR REPLACE INTO records
            (id, task_type, preconditions_json, canonical_steps_json,
             invariants_json, terminal_verifier, source_trajectories_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                d.get("id", ""), d.get("task_type", ""),
                json.dumps(d.get("candidate_preconditions", d.get("preconditions", {})),
                           ensure_ascii=False),
                json.dumps(d.get("param_steps", d.get("canonical_steps", [])),
                           ensure_ascii=False),
                json.dumps(d.get("invariants", []), ensure_ascii=False),
                d.get("terminal_verifier", ""),
                json.dumps(d.get("source_trajectory_ids", []), ensure_ascii=False),
                d.get("created_at", time.time()),
            ),
        )
        self.conn.commit()

    def load_records(self, task_type: str = "") -> list[dict]:
        """查询经验记录。"""
        sql = "SELECT * FROM records"
        params: list = []
        if task_type:
            sql += " WHERE task_type = ?"
            params.append(task_type)
        sql += " ORDER BY created_at"
        rows = self.conn.execute(sql, params).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM records LIMIT 0").description]
        return [dict(zip(cols, row)) for row in rows]

    def delete_record(self, record_id: str) -> None:
        """删除经验记录。"""
        self.conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        self.conn.commit()

    # ==================================================================
    # 向量存储
    # ==================================================================
    def save_embedding(
        self, text: str, vec: list[float], model_name: str = ""
    ) -> None:
        """持久化文本的 embedding 向量。"""
        import hashlib

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        self.conn.execute(
            """INSERT OR REPLACE INTO embeddings
            (text_hash, source_text, embedding, dim, model_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (text_hash, text[:500], _pack_vector(vec), len(vec), model_name, time.time()),
        )
        self.conn.commit()

    def get_embedding(self, text: str) -> list[float] | None:
        """获取已缓存的 embedding。"""
        import hashlib

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        row = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE text_hash = ?", (text_hash,)
        ).fetchone()
        if row and row[0]:
            return _unpack_vector(row[0])
        return None

    # ==================================================================
    # 环境 metadata
    # ==================================================================
    def save_env_metadata(self, info: dict) -> None:
        """保存环境信息到结构化列。"""
        os_info = info.get("os", {})
        py_info = info.get("python", {})
        hw = info.get("hardware", {})
        llm = info.get("llm_backends", {})
        ollama = llm.get("ollama", {})
        tau2 = info.get("tau2", {})
        env_vars = info.get("environment_variables", {})

        gpu = hw.get("gpus", [{}])
        gpu0 = gpu[0] if gpu else {}

        self.conn.execute(
            """INSERT INTO env_metadata
            (collected_at, os_system, os_release, os_distro, os_machine,
             python_version, python_executable, in_venv,
             cpu_count, memory_gb, gpu_name, gpu_memory,
             ollama_models, local_model_names,
             tau2_installed, tau2_version, tau2_domains,
             packages_json, eos_llm_backend, eos_ollama_model, eos_deepinfra_model)
            VALUES (?,?,?,?,?,?,?,? ,?,?,?,?,? ,?,?,? ,?,?,?,?,?)""",
            (
                info.get("collected_at", ""),
                os_info.get("system", ""), os_info.get("release", ""),
                os_info.get("distro", ""), os_info.get("machine", ""),
                py_info.get("version", "").split()[0] if py_info.get("version") else "",
                py_info.get("executable", ""),
                int(py_info.get("in_venv", False)),
                hw.get("cpu_count", 0), hw.get("memory_gb", 0),
                gpu0.get("name", ""), gpu0.get("memory", ""),
                ",".join(ollama.get("models", [])),
                ",".join(m["name"] for m in info.get("local_models", [])),
                int(tau2.get("installed", False)),
                tau2.get("version", ""),
                ",".join(tau2.get("domains", [])),
                json.dumps(info.get("packages", {}), ensure_ascii=False),
                env_vars.get("EOS_LLM_BACKEND", ""),
                env_vars.get("EOS_OLLAMA_MODEL", ""),
                env_vars.get("EOS_DEEPINFRA_MODEL", ""),
            ),
        )
        self.conn.commit()

    def get_latest_env_metadata(self) -> dict | None:
        """获取最新的环境信息（从结构化列重建）。"""
        row = self.conn.execute("SELECT * FROM env_metadata ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        cols = [d[0] for d in self.conn.execute("SELECT * FROM env_metadata LIMIT 0").description]
        d = dict(zip(cols, row))
        return {
            "collected_at": d["collected_at"],
            "os": {
                "system": d["os_system"], "release": d["os_release"],
                "distro": d["os_distro"], "machine": d["os_machine"],
            },
            "python": {
                "version": d["python_version"],
                "executable": d["python_executable"],
                "in_venv": bool(d["in_venv"]),
            },
            "hardware": {
                "cpu_count": d["cpu_count"], "memory_gb": d["memory_gb"],
                "gpus": [{"name": d["gpu_name"], "memory": d["gpu_memory"]}] if d["gpu_name"] else [],
            },
            "ollama_models": d["ollama_models"].split(",") if d["ollama_models"] else [],
            "local_models": d["local_model_names"].split(",") if d["local_model_names"] else [],
            "tau2": {
                "installed": bool(d["tau2_installed"]),
                "version": d["tau2_version"],
                "domains": d["tau2_domains"].split(",") if d["tau2_domains"] else [],
            },
            "packages": json.loads(d["packages_json"]) if d["packages_json"] else {},
            "eos_llm_backend": d["eos_llm_backend"],
        }

    # ==================================================================
    # 统计
    # ==================================================================
    def save_stats(self, task_type: str, stats: dict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO stats (task_type, stats_json, updated_at)
            VALUES (?, ?, ?)""",
            (task_type, json.dumps(stats, ensure_ascii=False), time.time()),
        )
        self.conn.commit()

    def load_stats(self, task_type: str) -> dict | None:
        row = self.conn.execute(
            "SELECT stats_json FROM stats WHERE task_type = ?", (task_type,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def load_all_stats(self) -> list[dict]:
        """查询所有 task_type 的统计（返回含 task_type 和 stats_json 的行）。"""
        rows = self.conn.execute("SELECT task_type, stats_json FROM stats").fetchall()
        return [{"task_type": r[0], "stats_json": r[1]} for r in rows]

    # ==================================================================
    # 迁移：从 JSON 文件导入
    # ==================================================================
    def migrate_from_json(self, data_dir: Path | None = None) -> dict:
        """从现有 JSON 文件迁移数据到 SQLite。

        Returns:
            迁移统计 {trajectories: N, records: N, harnesses: N, stats: N}
        """
        if data_dir is None:
            data_dir = self.config.data_dir

        from experience_os.models import (
            ExperienceRecord, Harness, HarnessStatus,
            TaskTypeStats, Trajectory, VerificationMeta,
        )
        from experience_os.repository import _as_dict

        stats = {"trajectories": 0, "records": 0, "harnesses": 0, "stats_s": 0}

        # 迁移轨迹
        for p in (data_dir / "trajectories").glob("*.json"):
            try:
                self.save_trajectory(json.loads(p.read_text()))
                stats["trajectories"] += 1
            except Exception as exc:
                log.warning("Failed to migrate trajectory %s: %s", p.stem, exc)

        # 迁移记录
        for p in (data_dir / "records").glob("*.json"):
            try:
                d = json.loads(p.read_text())
                self.conn.execute(
                    """INSERT OR REPLACE INTO records
                    (id, task_type, preconditions_json, canonical_steps_json,
                     invariants_json, terminal_verifier, source_trajectories_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        d.get("id", p.stem), d.get("task_type", ""),
                        json.dumps(d.get("preconditions", {}), ensure_ascii=False),
                        json.dumps(d.get("param_steps", []), ensure_ascii=False),
                        json.dumps(d.get("invariants", []), ensure_ascii=False),
                        d.get("terminal_verifier", ""),
                        json.dumps(d.get("source_trajectory_ids", []), ensure_ascii=False),
                        d.get("created_at", time.time()),
                    ),
                )
                stats["records"] += 1
            except Exception as exc:
                log.warning("Failed to migrate record %s: %s", p.stem, exc)

        # 迁移 Harness
        for p in (data_dir / "harnesses").glob("*.json"):
            try:
                d = json.loads(p.read_text())
                embed = d.pop("embedding", None)
                self.save_harness(d, embedding=embed)
                stats["harnesses"] += 1
            except Exception as exc:
                log.warning("Failed to migrate harness %s: %s", p.stem, exc)

        # 迁移统计
        for p in (data_dir / "stats").glob("*.json"):
            try:
                d = json.loads(p.read_text())
                self.save_stats(d.get("task_type", p.stem), d)
                stats["stats_s"] += 1
            except Exception as exc:
                log.warning("Failed to migrate stats %s: %s", p.stem, exc)

        self.conn.commit()
        log.info("Migration from JSON complete: %s", stats)
        return stats

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
