"""LTS 经验库：持久底座，保存全部原始交互/结果。

设计原则：
    * **append-only** — 只追加，永不修改/删除，任何归纳方案都不影响底座
    * **与归纳解耦** — LTS 只记录"发生了什么"，不关心是否被归纳利用
    * **跨实验复用** — 多次实验共享同一 LTS，按 experiment_id 隔离查询
    * **成本收敛可查** — 提供按任务序号的累计 token / 滚动成本曲线接口

位于 Repository/Storage 四层存储之下，是最低层的"原始经验"。
上层的归纳（无论何种 artifact 形态）都从 LTS 读取素材，互不影响。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_LTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS lts_log (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at   REAL NOT NULL,
    experiment_id TEXT NOT NULL,
    method      TEXT NOT NULL,
    domain      TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    task_type   TEXT NOT NULL,
    idx         INTEGER,          -- 任务在流中的序号
    phase       TEXT,             -- warmup | eval
    success     INTEGER,          -- 0/1
    reward      REAL,
    tokens      INTEGER,
    latency     REAL,
    path        TEXT,             -- agent | harness | harness+agent
    trajectory_json TEXT,         -- 完整轨迹（可选）
    meta_json   TEXT              -- 额外元数据
);
CREATE INDEX IF NOT EXISTS idx_lts_exp   ON lts_log(experiment_id);
CREATE INDEX IF NOT EXISTS idx_lts_type ON lts_log(task_type);
CREATE INDEX IF NOT EXISTS idx_lts_meth ON lts_log(method);
CREATE INDEX IF NOT EXISTS idx_lts_succ ON lts_log(success);
"""


@dataclass
class LTSEntry:
    """单次执行事件的原始记录。"""
    experiment_id: str
    method: str
    domain: str
    task_id: str
    task_type: str
    idx: int = 0
    phase: str = ""
    success: bool = False
    reward: float = 0.0
    tokens: int = 0
    latency: float = 0.0
    path: str = "agent"
    trajectory: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class LTSStore:
    """Append-only 持久经验底座。

    单个 SQLite 文件，跨实验共享。每个实验用 ``experiment_id`` 隔离。
    """

    def __init__(self, db_path: str | Path = ".experience_os_data/lts.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript(_LTS_SCHEMA)
        conn.commit()
        self._conn = conn
        log.info("LTS store initialized at %s", self.db_path)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def log(self, entry: LTSEntry) -> int:
        """追加一条执行记录，返回 seq。"""
        cols = ("logged_at, experiment_id, method, domain, task_id, task_type, "
                "idx, phase, success, reward, tokens, latency, path, "
                "trajectory_json, meta_json")
        placeholders = ",".join(["?"] * 15)
        c = self.conn.execute(
            f"INSERT INTO lts_log ({cols}) VALUES ({placeholders})",
            (
                time.time(), entry.experiment_id, entry.method, entry.domain,
                entry.task_id, entry.task_type, entry.idx, entry.phase,
                int(entry.success), entry.reward, entry.tokens, entry.latency,
                entry.path,
                json.dumps(entry.trajectory, ensure_ascii=False) if entry.trajectory else None,
                json.dumps(entry.meta, ensure_ascii=False) if entry.meta else None,
            ),
        )
        self.conn.commit()
        return int(c.lastrowid)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def query(
        self,
        *,
        experiment_id: str = "",
        method: str = "",
        domain: str = "",
        task_type: str = "",
        success_only: bool = False,
    ) -> list[dict]:
        """按条件查询原始记录。"""
        sql = "SELECT * FROM lts_log WHERE 1=1"
        params: list = []
        if experiment_id:
            sql += " AND experiment_id=?"
            params.append(experiment_id)
        if method:
            sql += " AND method=?"
            params.append(method)
        if domain:
            sql += " AND domain=?"
            params.append(domain)
        if task_type:
            sql += " AND task_type=?"
            params.append(task_type)
        if success_only:
            sql += " AND success=1"
        sql += " ORDER BY seq"
        rows = self.conn.execute(sql, params).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM lts_log LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    # ------------------------------------------------------------------
    # 成本收敛接口
    # ------------------------------------------------------------------
    def cost_curve(
        self,
        experiment_id: str,
        *,
        window: int = 3,
    ) -> dict:
        """返回单实验的成本收敛曲线数据。

        含滚动成功率、累计 token、滚动平均 token。
        """
        rows = self.query(experiment_id=experiment_id)
        if not rows:
            return {"x": [], "rolling_sr": [], "cumulative_tokens": [], "rolling_avg_tokens": []}
        successes = [bool(r["success"]) for r in rows]
        tokens = [int(r["tokens"] or 0) for r in rows]

        def _rolling(vals: list, w: int) -> list:
            out = []
            for i in range(len(vals)):
                lo = max(0, i - w + 1)
                chunk = vals[lo: i + 1]
                out.append(sum(chunk) / len(chunk))
            return out

        cum = []
        s = 0
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
        """列出所有实验及其汇总。"""
        rows = self.conn.execute(
            """SELECT experiment_id, method, domain, COUNT(*) as n,
                      SUM(success) as ok, SUM(tokens) as tok
               FROM lts_log GROUP BY experiment_id ORDER BY MIN(seq)"""
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
