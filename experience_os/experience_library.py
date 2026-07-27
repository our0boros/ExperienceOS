"""SQLite implementation shared by the Trace, Experience, and Artifact stores.

.. deprecated::
    新代码应通过 ``stores.py`` 的 Store facade 访问：
    ``stores_for(library)`` → (TraceStore, ExperienceStore, ArtifactStore)。
    直接调用 ``ExperienceLibrary`` 的方法仅保留向后兼容。

原始轨迹 append-only，历史 SQLite 数据和迁移能力必须保留。

三层结构：

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
    method          TEXT NOT NULL,          -- vanilla | react | coe | skillopt
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

-- ========== 底层：子步骤（独立一等实体） ==========
CREATE TABLE IF NOT EXISTS substeps (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id   TEXT NOT NULL,         -- FK to trajectories.task_id
    experiment_id   TEXT NOT NULL,
    plan_idx        INTEGER NOT NULL,      -- 在 Plan 中的位置（0-based）
    intent          TEXT NOT NULL,         -- "lookup_user_by_email"
    tool_name       TEXT NOT NULL,         -- "find_user_id_by_email"
    -- 三要素检索签名
    input_schema    TEXT,                  -- JSON: {"requires":["email"],"from":"task"}
    output_schema   TEXT,                  -- JSON: {"produces":["user_id"],"type":"str"}
    effect          TEXT DEFAULT 'read_only', -- read_only | write | mixed
    -- 执行记录
    params_json     TEXT,
    result_json     TEXT,
    success         INTEGER DEFAULT 0,
    execution_mode  TEXT DEFAULT 'agent',  -- agent | harness | plan
    tokens_used     INTEGER DEFAULT 0,
    -- 全路径历史
    artifact_id     TEXT,                  -- 使用的 harness ID（如有）
    artifact_version INTEGER,
    failure_type    TEXT,                  -- F1-F4（harness 失败时）
    source          TEXT DEFAULT 'react',  -- react | plan_execute | harness | llm_glue
    meta_json       TEXT,                  -- plan JSON / glue code / decompose prompt 等
    -- 父任务上下文（用于贝叶斯权重）
    parent_task_type     TEXT,
    parent_task_success  INTEGER DEFAULT 0,
    -- 向量
    intent_embedding BLOB,                 -- float32 向量
    embedding_model  TEXT,
    logged_at        REAL
);
CREATE INDEX IF NOT EXISTS idx_substeps_intent   ON substeps(intent);
CREATE INDEX IF NOT EXISTS idx_substeps_tool     ON substeps(tool_name, experiment_id);
CREATE INDEX IF NOT EXISTS idx_substeps_parent   ON substeps(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_substeps_io       ON substeps(effect, input_schema);
CREATE INDEX IF NOT EXISTS idx_substeps_artifact ON substeps(artifact_id);
CREATE INDEX IF NOT EXISTS idx_substeps_source   ON substeps(source);

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


@dataclass
class SubStepRecord:
    """子步骤的持久化记录（对应 ``substeps`` 表）。

    子步骤是一等实体：独立于全任务结果，可跨任务类型检索。
    """
    trajectory_id: str
    experiment_id: str
    plan_idx: int
    intent: str
    tool_name: str
    input_schema: str = ""
    output_schema: str = ""
    effect: str = "read_only"
    params_json: str = ""
    result_json: str = ""
    success: bool = False
    execution_mode: str = "agent"
    tokens_used: int = 0
    artifact_id: str = ""
    artifact_version: int = 0
    failure_type: str = ""
    source: str = "react"
    meta_json: str = ""
    parent_task_type: str = ""
    parent_task_success: bool = False
    intent_embedding: Optional[bytes] = None
    embedding_model: str = ""
    logged_at: float = 0.0
    # 预测契约（Phase A，来自 flow.md 融合）
    # 存储于 meta_json 中以避免 schema 变更；此处为便利访问字段
    prediction_accuracy: float = 1.0   # 0.0–1.0，默认 1.0（未启用预测契约时）
    quality_label: str = ""             # high_quality | lucky_success | implementation_defect | negative_sample


def serialize_messages(messages: list[Any]) -> str:
    """将 tau2 消息对象序列化为完整 JSON（保留 prompt 和回复全文）。

    每条消息记录 role / content / tool_calls / tool_results / usage。
    usage 字段包含 API 返回的实际 prompt_tokens 和 completion_tokens。
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
        # API token usage (preserved for accurate cost tracking)
        usage = getattr(msg, "usage", None)
        if usage:
            entry["usage"] = dict(usage) if not isinstance(usage, dict) else usage
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
    # 底层：子步骤（独立一等实体）
    # ==================================================================
    def log_substep(self, rec: SubStepRecord) -> int:
        """写入单条子步骤记录。"""
        intent_blob: Optional[bytes] = None
        if rec.intent_embedding is not None:
            intent_blob = rec.intent_embedding

        # 预测契约字段编码到 meta_json（Phase A，无 schema 变更）
        meta: dict[str, Any] = {}
        if rec.meta_json:
            try:
                meta = json.loads(rec.meta_json)
            except Exception:
                meta = {"raw": rec.meta_json}
        if rec.prediction_accuracy != 1.0 or rec.quality_label:
            meta["prediction_accuracy"] = rec.prediction_accuracy
            meta["quality_label"] = rec.quality_label
        meta_json = json.dumps(meta, ensure_ascii=False) if meta else rec.meta_json

        c = self.conn.execute(
            """INSERT INTO substeps
               (trajectory_id, experiment_id, plan_idx, intent, tool_name,
                input_schema, output_schema, effect,
                params_json, result_json, success, execution_mode, tokens_used,
                artifact_id, artifact_version, failure_type, source, meta_json,
                parent_task_type, parent_task_success,
                intent_embedding, embedding_model, logged_at)
               VALUES (?,?,?,?,?, ?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?, ?,?,?)""",
            (
                rec.trajectory_id, rec.experiment_id, rec.plan_idx,
                rec.intent, rec.tool_name,
                rec.input_schema, rec.output_schema, rec.effect,
                rec.params_json, rec.result_json,
                int(rec.success), rec.execution_mode, rec.tokens_used,
                rec.artifact_id, rec.artifact_version,
                rec.failure_type, rec.source, meta_json,
                rec.parent_task_type, int(rec.parent_task_success),
                intent_blob, rec.embedding_model, time.time(),
            ),
        )
        self.conn.commit()
        return c.lastrowid

    def log_substeps_batch(self, records: list[SubStepRecord]) -> list[int]:
        """批量写入子步骤记录。"""
        ids = []
        for rec in records:
            ids.append(self.log_substep(rec))
        return ids

    def query_substeps_by_intent(
        self, intent: str, *, experiment_id: str = "", limit: int = 100,
    ) -> list[dict]:
        """按意图查询子步骤记录。"""
        sql = "SELECT * FROM substeps WHERE intent=?"
        params: list = [intent]
        if experiment_id:
            sql += " AND experiment_id=?"
            params.append(experiment_id)
        sql += " ORDER BY logged_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        col_names = [d[0] for d in self.conn.execute(
            "SELECT * FROM substeps LIMIT 0").description]
        return [dict(zip(col_names, r)) for r in rows]

    def aggregate_substep_patterns(
        self, *, experiment_id: str = "", min_support: int = 0,
    ) -> list[dict]:
        """从 substeps 表聚合子步骤模式（贝叶斯权重 + 预测契约调整）。

        返回每个 ``(intent, tool_name)`` 组合的统计：
        - support_count: 去重轨迹数
        - success_count: 单步成功的执行次数
        - success_in_full_tasks: 在成功全任务中的出现次数
        - total_appearances: 在任何全任务中的出现次数
        - bayesian_score: 贝叶斯可信度（α=1, β=1）
        - adjusted_bayesian: 经预测准确率调整后的评分（Phase A）
        - prediction_accuracy: 平均预测准确率
        - quality_label_dist: 质量标签分布
        """
        sql = """SELECT
            intent, tool_name, effect,
            COUNT(DISTINCT trajectory_id) AS support_count,
            SUM(success) AS success_count,
            SUM(CASE WHEN parent_task_success=1 THEN 1 ELSE 0 END) AS success_in_full_tasks,
            COUNT(*) AS total_appearances,
            COUNT(DISTINCT trajectory_id) as total_trajectories
        FROM substeps
        WHERE 1=1"""
        params: list = []
        if experiment_id:
            sql += " AND experiment_id=?"
            params.append(experiment_id)
        sql += """ GROUP BY intent, tool_name
        HAVING support_count >= ?
        ORDER BY support_count DESC"""
        params.append(min_support if min_support > 0 else 1)

        rows = self.conn.execute(sql, params).fetchall()

        # Phase A: 为每个分组额外查询 meta_json 中的预测数据
        pred_sql = """SELECT meta_json, success, parent_task_success
                      FROM substeps WHERE intent=? AND tool_name=?"""
        if experiment_id:
            pred_sql += " AND experiment_id=?"
        pred_params_base: list = []
        if experiment_id:
            pred_params_base.append(experiment_id)

        results = []
        for row in rows:
            intent = row[0] if isinstance(row, tuple) else row.get("intent", "")
            tool_name = row[1] if isinstance(row, tuple) else row.get("tool_name", "")
            effect = row[2] if isinstance(row, tuple) else row.get("effect", "")
            sc = (row[3] if isinstance(row, tuple) else row.get("support_count", 0)) or 0
            succ = (row[4] if isinstance(row, tuple) else row.get("success_count", 0)) or 0
            sft = (row[5] if isinstance(row, tuple) else row.get("success_in_full_tasks", 0)) or 0
            ta = (row[6] if isinstance(row, tuple) else row.get("total_appearances", 0)) or 0

            success_rate = succ / ta if ta > 0 else 0.0
            alpha, beta_val = 1.0, 1.0
            bayesian_score = (alpha + sft) / (alpha + beta_val + ta) if (alpha + beta_val + ta) > 0 else 0.0

            # Phase A: 收集预测准确率
            pred_params = [intent, tool_name] + pred_params_base
            meta_rows = self.conn.execute(pred_sql, pred_params).fetchall()
            pred_accs = []
            qual_labels = []
            for (meta_json, _, _) in meta_rows:
                pred_acc = 1.0
                qual = ""
                if meta_json:
                    try:
                        meta = json.loads(meta_json)
                        if isinstance(meta, dict):
                            pred_acc = float(meta.get("prediction_accuracy", 1.0))
                            qual = meta.get("quality_label", "")
                    except Exception:
                        pass
                pred_accs.append(pred_acc)
                if qual:
                    qual_labels.append(qual)

            avg_pred_acc = sum(pred_accs) / len(pred_accs) if pred_accs else 1.0
            pred_multiplier = 0.5 + 0.5 * avg_pred_acc
            adjusted_bayesian = bayesian_score * pred_multiplier

            # 侥幸成功惩罚
            from collections import Counter
            label_counts = Counter(qual_labels)
            lucky_ratio = label_counts.get("lucky_success", 0) / max(1, len(pred_accs))
            if lucky_ratio > 0.3:
                adjusted_bayesian *= 0.7

            d = {
                "intent": intent,
                "tool_name": tool_name,
                "action_name": tool_name,
                "effect": effect or "read_only",
                "support_count": sc,
                "success_count": succ,
                "success_rate": success_rate,
                "success_in_full_tasks": sft,
                "total_appearances": ta,
                "bayesian_score": bayesian_score,
                "adjusted_bayesian": adjusted_bayesian,
                "score": adjusted_bayesian,
                "prediction_accuracy": avg_pred_acc,
                "quality_label_dist": dict(label_counts),
            }
            results.append(d)

        return results

    def migrate_substeps_from_trajectories(self, experiment_id: str = "") -> int:
        """从已有轨迹的 steps_json 字段回填 substeps 表。

        遍历所有轨迹，解析其 steps_json（如有），为每个 step 创建
        一条 SubStepRecord。已存在的记录（相同 trajectory_id + plan_idx）跳过。

        Returns: 新创建的子步骤数量。
        """
        import json as _json

        sql = "SELECT task_id, experiment_id, task_type, success, steps_json FROM trajectories"
        params: list = []
        if experiment_id:
            sql += " WHERE experiment_id=?"
            params.append(experiment_id)

        created = 0
        for row in self.conn.execute(sql, params).fetchall():
            task_id, exp_id, task_type, task_success, steps_json_str = row
            if not steps_json_str:
                continue
            try:
                steps = _json.loads(steps_json_str) if isinstance(steps_json_str, str) else steps_json_str
            except Exception:
                continue

            if not isinstance(steps, list):
                continue

            for i, step in enumerate(steps):
                # Skip if already exists
                existing = self.conn.execute(
                    "SELECT seq FROM substeps WHERE trajectory_id=? AND plan_idx=?",
                    (task_id, i),
                ).fetchone()
                if existing:
                    continue

                action = step.get("action", "") if isinstance(step, dict) else getattr(step, "action", "")
                tool_name = action.split("(")[0].strip() if action else ""
                intent = (step.get("sub_step_intent", "") or tool_name) if isinstance(step, dict) else (
                    getattr(step, "sub_step_intent", "") or tool_name
                )

                rec = SubStepRecord(
                    trajectory_id=task_id,
                    experiment_id=exp_id,
                    plan_idx=i,
                    intent=intent,
                    tool_name=tool_name,
                    success=bool(step.get("result", "")) and "Error" not in str(step.get("result", ""))
                        if isinstance(step, dict) else False,
                    execution_mode="agent",
                    parent_task_type=task_type,
                    parent_task_success=bool(task_success),
                )
                self.log_substep(rec)
                created += 1

        log.info("Substep migration: %d records created", created)
        return created

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
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
