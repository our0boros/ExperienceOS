"""层级化经验库（SQLite）。

三层结构（对应 STRUCTURE.md §5）：

    底层 trajectories  — 完整轨迹：任务内容、完整对话（LLM 看到的 prompt
                          和回复）、tool calls/results、reward、tokens。
                          **append-only，永不删除**。
    中层 records       — 经验记录：前置条件、参数化步骤、不变量。版本化。
    上层 artifacts     — harnesses/skills：可执行/文本 artifact。版本 DAG。

多实例：
    * **LTS 库**（``.experience_os_data/lts_library.db``）— 持久，底层 trajs
      永不丢失，上层随版本更新优化总结。
    * **实验库**（``.experience_os_data/exp_<id>.db``）— 临时，服务于单次实验，
      实验结束可丢弃。原始数据仍在 LTS 库。

与归纳方案解耦：无论上层用什么归纳算法（code/text/AST），都从底层 trajs
读取素材，互不影响。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


_SCHEMA = """
-- ========== 底层：完整轨迹（append-only） ==========
CREATE TABLE IF NOT EXISTS trajectories (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at       REAL NOT NULL,
    experiment_id   TEXT NOT NULL,
    method          TEXT NOT NULL,          -- vanilla | react | autoharness | skillopt
    domain          TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    task_type       TEXT NOT NULL,
    task_description TEXT,
    idx             INTEGER,                -- 任务在流中的序号
    phase           TEXT,                   -- warmup | eval
    success         INTEGER,                -- 0/1
    reward          REAL,
    tokens          INTEGER,
    latency         REAL,
    path            TEXT,                   -- agent | harness | harness+agent
    -- 完整内容（核心价值）
    task_json       TEXT,                   -- 完整任务对象
    messages_json   TEXT,                   -- 完整对话：每条消息 role/content/tool_calls/results
    steps_json      TEXT,                   -- 结构化步骤（从 messages 提取）
    meta_json       TEXT                    -- 额外元数据
);
CREATE INDEX IF NOT EXISTS idx_traj_exp  ON trajectories(experiment_id);
CREATE INDEX IF NOT EXISTS idx_traj_type ON trajectories(task_type);
CREATE INDEX IF NOT EXISTS idx_traj_succ ON trajectories(success);
CREATE INDEX IF NOT EXISTS idx_traj_meth ON trajectories(method);

-- ========== 中层：经验记录（版本化） ==========
CREATE TABLE IF NOT EXISTS records (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      REAL NOT NULL,
    experiment_id   TEXT,
    task_type       TEXT NOT NULL,
    preconditions_json TEXT,
    param_steps_json  TEXT,
    invariants_json   TEXT,
    terminal_verifier TEXT,
    source_trajectory_ids_json TEXT,
    support_count   INTEGER,
    version         INTEGER DEFAULT 1,
    superseded_by   INTEGER,                -- 指向更新版本（NULL=当前）
    meta_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_rec_type ON records(task_type);

-- ========== 上层：artifacts（版本 DAG） ==========
CREATE TABLE IF NOT EXISTS artifacts (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      REAL NOT NULL,
    experiment_id   TEXT,
    task_type       TEXT NOT NULL,
    artifact_type   TEXT NOT NULL,          -- harness | skill
    name            TEXT,
    procedure_code  TEXT,                   -- 可执行代码（harness）
    skill_text      TEXT,                   -- 文本技能（skill）
    verification_status TEXT,               -- draft | verified | needs_revision | deprecated
    validation_score REAL,
    embedding_blob  BLOB,
    parent_seq      INTEGER,                -- DAG 父节点（patch/specialization/composition）
    edge_type       TEXT,                   -- patch | specialization | composition
    version         TEXT,
    meta_json       TEXT,
    FOREIGN KEY (parent_seq) REFERENCES artifacts(seq)
);
CREATE INDEX IF NOT EXISTS idx_art_type ON artifacts(task_type);
CREATE INDEX IF NOT EXISTS idx_art_stat ON artifacts(verification_status);

-- ========== 统计 ==========
CREATE TABLE IF NOT EXISTS stats (
    task_type           TEXT PRIMARY KEY,
    total_executions    INTEGER DEFAULT 0,
    agent_executions    INTEGER DEFAULT 0,
    harness_executions  INTEGER DEFAULT 0,
    agent_successes     INTEGER DEFAULT 0,
    harness_successes   INTEGER DEFAULT 0,
    failure_counts_json TEXT,
    estimated_token_savings INTEGER DEFAULT 0,
    updated_at          REAL
);

-- ========== embedding 缓存 ==========
CREATE TABLE IF NOT EXISTS embeddings (
    text_hash   TEXT PRIMARY KEY,
    embedding   BLOB,
    model       TEXT,
    created_at  REAL
);
"""


@dataclass
class TrajectoryRecord:
    """底层轨迹的完整记录。"""
    experiment_id: str
    method: str
    domain: str
    task_id: str
    task_type: str
    task_description: str = ""
    idx: int = 0
    phase: str = ""
    success: bool = False
    reward: float = 0.0
    tokens: int = 0
    latency: float = 0.0
    path: str = "agent"
    task_json: str = ""
    messages_json: str = ""  # 完整对话
    steps_json: str = ""
    meta: dict = field(default_factory=dict)


def serialize_messages(messages: list[Any]) -> str:
    """将 tau2 消息对象序列化为完整 JSON（保留 prompt 和回复全文）。

    每条消息记录 role / content / tool_calls / tool_results。
    """
    out = []
    for msg in messages:
        entry: dict[str, Any] = {"role": getattr(msg, "role", type(msg).__name__)}
        content = getattr(msg, "content", None)
        if content is not None:
            entry["content"] = str(content)
        # tool calls (assistant)
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            entry["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in tcs
            ]
        # tool results (tool messages)
        tool_msgs = getattr(msg, "tool_messages", None)
        if tool_msgs:
            entry["tool_results"] = [str(tm.content)[:2000] for tm in tool_msgs]
        elif getattr(msg, "tool_call_id", None):
            entry["tool_call_id"] = str(msg.tool_call_id)
        out.append(entry)
    return json.dumps(out, ensure_ascii=False, default=str)


class ExperienceLibrary:
    """层级化经验库。

    用法::

        # LTS 持久库
        lts = ExperienceLibrary.persistent()
        # 实验临时库
        exp_lib = ExperienceLibrary.experiment("my-exp-001")
    """

    def __init__(self, db_path: str | Path, *, persistent: bool = False) -> None:
        self.db_path = Path(db_path)
        self.persistent = persistent
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    @classmethod
    def persistent(cls, path: str | Path = ".experience_os_data/lts_library.db") -> "ExperienceLibrary":
        return cls(path, persistent=True)

    @classmethod
    def experiment(cls, experiment_id: str) -> "ExperienceLibrary":
        return cls(f".experience_os_data/exp_{experiment_id}.db", persistent=False)

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn
        tag = "LTS(persistent)" if self.persistent else "experiment"
        log.info("ExperienceLibrary initialized [%s]: %s", tag, self.db_path)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    # ==================================================================
    # 底层：轨迹（append-only）
    # ==================================================================
    def log_trajectory(self, rec: TrajectoryRecord) -> int:
        """追加一条完整轨迹记录，返回 seq。永不修改/删除。"""
        cols = ("experiment_id, method, domain, task_id, task_type, "
                "task_description, idx, phase, success, reward, tokens, "
                "latency, path, task_json, messages_json, steps_json, meta_json, logged_at")
        placeholders = ",".join(["?"] * 18)
        c = self.conn.execute(
            f"INSERT INTO trajectories ({cols}) VALUES ({placeholders})",
            (
                rec.experiment_id, rec.method, rec.domain, rec.task_id,
                rec.task_type, rec.task_description, rec.idx, rec.phase,
                int(rec.success), rec.reward, rec.tokens, rec.latency,
                rec.path, rec.task_json, rec.messages_json, rec.steps_json,
                json.dumps(rec.meta, ensure_ascii=False) if rec.meta else None,
                time.time(),
            ),
        )
        self.conn.commit()
        return int(c.lastrowid)

    def query_trajectories(
        self, *, experiment_id: str = "", task_type: str = "",
        success_only: bool = False, domain: str = "", method: str = "",
        with_messages: bool = False,
    ) -> list[dict]:
        cols = ("seq, experiment_id, method, domain, task_id, task_type, "
                "task_description, idx, phase, success, reward, tokens, "
                "latency, path")
        if with_messages:
            cols += ", task_json, messages_json, steps_json, meta_json"
        sql = f"SELECT {cols} FROM trajectories WHERE 1=1"
        params: list = []
        if experiment_id:
            sql += " AND experiment_id=?"
            params.append(experiment_id)
        if task_type:
            sql += " AND task_type=?"
            params.append(task_type)
        if domain:
            sql += " AND domain=?"
            params.append(domain)
        if method:
            sql += " AND method=?"
            params.append(method)
        if success_only:
            sql += " AND success=1"
        sql += " ORDER BY seq"
        rows = self.conn.execute(sql, params).fetchall()
        col_names = [d[0] for d in self.conn.execute(
            f"SELECT {cols} FROM trajectories LIMIT 0").description]
        return [dict(zip(col_names, r)) for r in rows]

    # ==================================================================
    # 中层：经验记录
    # ==================================================================
    def log_record(
        self, *, task_type: str, preconditions: dict, param_steps: list,
        invariants: list, terminal_verifier: str = "",
        source_ids: list = None, support_count: int = 0,
        experiment_id: str = "", meta: dict = None,
    ) -> int:
        c = self.conn.execute(
            """INSERT INTO records
               (created_at, experiment_id, task_type, preconditions_json,
                param_steps_json, invariants_json, terminal_verifier,
                source_trajectory_ids_json, support_count, meta_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), experiment_id, task_type,
                json.dumps(preconditions, ensure_ascii=False),
                json.dumps(param_steps, ensure_ascii=False),
                json.dumps(invariants, ensure_ascii=False),
                terminal_verifier,
                json.dumps(source_ids or [], ensure_ascii=False),
                support_count,
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return int(c.lastrowid)

    def get_records(self, task_type: str, current_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM records WHERE task_type=?"
        if current_only:
            sql += " AND superseded_by IS NULL"
        sql += " ORDER BY seq DESC"
        rows = self.conn.execute(sql, [task_type]).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM records LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    # ==================================================================
    # 上层：artifacts
    # ==================================================================
    def log_artifact(
        self, *, task_type: str, artifact_type: str = "harness",
        name: str = "", procedure_code: str = "", skill_text: str = "",
        verification_status: str = "draft", validation_score: float = 0.0,
        embedding_blob: bytes = None, parent_seq: int = None,
        edge_type: str = "", version: str = "", experiment_id: str = "",
        meta: dict = None,
    ) -> int:
        c = self.conn.execute(
            """INSERT INTO artifacts
               (created_at, experiment_id, task_type, artifact_type, name,
                procedure_code, skill_text, verification_status, validation_score,
                embedding_blob, parent_seq, edge_type, version, meta_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), experiment_id, task_type, artifact_type, name,
                procedure_code, skill_text, verification_status, validation_score,
                embedding_blob, parent_seq, edge_type, version,
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return int(c.lastrowid)

    def get_artifacts(
        self, *, task_type: str = "", artifact_type: str = "",
        status: str = "verified",
    ) -> list[dict]:
        sql = "SELECT * FROM artifacts WHERE 1=1"
        params: list = []
        if task_type:
            sql += " AND task_type=?"
            params.append(task_type)
        if artifact_type:
            sql += " AND artifact_type=?"
            params.append(artifact_type)
        if status:
            sql += " AND verification_status=?"
            params.append(status)
        sql += " ORDER BY seq DESC"
        rows = self.conn.execute(sql, params).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM artifacts LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    # ==================================================================
    # 统计
    # ==================================================================
    def update_stats(self, task_type: str, *, agent: bool, success: bool,
                     tokens: int) -> None:
        row = self.conn.execute(
            "SELECT * FROM stats WHERE task_type=?", [task_type]
        ).fetchone()
        if row is None:
            self.conn.execute(
                """INSERT INTO stats (task_type, total_executions, agent_executions,
                   harness_executions, agent_successes, harness_successes, updated_at)
                   VALUES (?,1,?,0,?,0,?)""",
                (task_type, int(agent), int(success), time.time()),
            )
        else:
            if agent:
                self.conn.execute(
                    "UPDATE stats SET total_executions=total_executions+1, "
                    "agent_executions=agent_executions+1, "
                    f"agent_successes=agent_successes+{int(success)}, updated_at=? "
                    "WHERE task_type=?",
                    (time.time(), task_type),
                )
            else:
                self.conn.execute(
                    "UPDATE stats SET total_executions=total_executions+1, "
                    "harness_executions=harness_executions+1, "
                    f"harness_successes=harness_successes+{int(success)}, updated_at=? "
                    "WHERE task_type=?",
                    (time.time(), task_type),
                )
        self.conn.commit()

    def get_stats(self, task_type: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM stats WHERE task_type=?", [task_type]
        ).fetchone()
        if not row:
            return {}
        cols = [d[0] for d in self.conn.execute("SELECT * FROM stats LIMIT 0").description]
        return dict(zip(cols, row))

    # ==================================================================
    # 成本收敛
    # ==================================================================
    def cost_curve(self, experiment_id: str, *, window: int = 3) -> dict:
        rows = self.query_trajectories(experiment_id=experiment_id)
        if not rows:
            return {"x": [], "rolling_sr": [], "cumulative_tokens": [],
                    "rolling_avg_tokens": []}
        successes = [bool(r["success"]) for r in rows]
        tokens = [int(r["tokens"] or 0) for r in rows]

        def _rolling(vals, w):
            out = []
            for i in range(len(vals)):
                lo = max(0, i - w + 1)
                chunk = vals[lo: i + 1]
                out.append(sum(chunk) / len(chunk))
            return out

        cum, s = [], 0
        for t in tokens:
            s += t
            cum.append(s)
        return {
            "x": list(range(1, len(rows) + 1)),
            "rolling_sr": _rolling([1 if x else 0 for x in successes], window),
            "cumulative_tokens": cum,
            "rolling_avg_tokens": _rolling(tokens, window),
            "total_tokens": s,
            "total_tasks": len(rows),
            "success_rate": sum(successes) / len(successes),
        }

    def experiments(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT experiment_id, method, domain, COUNT(*) as n,
                      SUM(success) as ok, SUM(tokens) as tok
               FROM trajectories GROUP BY experiment_id ORDER BY MIN(seq)"""
        ).fetchall()
        return [
            {"experiment_id": r[0], "method": r[1], "domain": r[2],
             "tasks": r[3], "successes": r[4] or 0, "tokens": r[5] or 0}
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
