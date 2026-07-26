"""一次性迁移：experience_os.db (legacy) → lts_library.db (LTS)。

迁移内容：
- embeddings: 直接复制 (同 schema)
- harnesses: ACTIVE 状态的 → artifacts (字段映射)
- records: → records (字段映射)
- stats: stats_json 解析 → 独立列

运行后不删除旧 DB — 手动确认后再删。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DATA = Path(".experience_os_data")
LEGACY = DATA / "experience_os.db"
LTS = DATA / "lts_library.db"


def migrate_embeddings(leg: sqlite3.Connection, lts: sqlite3.Connection) -> int:
    """复制 embeddings（跳过已存在的 text_hash）。"""
    existing = {r[0] for r in lts.execute("SELECT text_hash FROM embeddings")}
    rows = leg.execute("SELECT text_hash, source_text, embedding, dim, model_name, created_at FROM embeddings").fetchall()
    n = 0
    for text_hash, source_text, emb, dim, model, ts in rows:
        if text_hash in existing:
            continue
        try:
            lts.execute(
                "INSERT INTO embeddings (text_hash, embedding, model, created_at) VALUES (?, ?, ?, ?)",
                (text_hash, emb, model or "unknown", ts or time.time()),
            )
            n += 1
        except Exception:
            pass
    return n


def migrate_harnesses(leg: sqlite3.Connection, lts: sqlite3.Connection) -> int:
    """迁移 ACTIVE 状态的 harness → LTS artifacts。"""
    existing_names = {r[0] for r in lts.execute("SELECT name FROM artifacts")}
    rows = leg.execute("SELECT * FROM harnesses WHERE status = 'active'").fetchall()
    cols = [d[0] for d in leg.execute("PRAGMA table_info(harnesses)")]
    n = 0
    now = time.time()
    for row in rows:
        d = dict(zip(cols, row))
        name = d["id"]
        if name in existing_names:
            continue
        # 解析 verification_json 提取 validation_score
        score = 0.0
        try:
            v = json.loads(d.get("verification_json", "{}") or "{}")
            score = float(v.get("success_rate", v.get("validation_score", 0.0)))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        # 打包剩余字段到 meta_json
        meta = {
            "capability": d.get("capability", ""),
            "preconditions_json": d.get("preconditions_json", "{}"),
            "invariants_json": d.get("invariants_json", "[]"),
            "params_json": d.get("params_json", "[]"),
            "failure_counts_json": d.get("failure_counts_json", "{}"),
            "verification_json": d.get("verification_json", "{}"),
            "parent_id": d.get("parent_id", ""),
            "branch": d.get("branch", "main"),
            "updated_at": d.get("updated_at"),
        }
        try:
            lts.execute(
                """INSERT INTO artifacts
                   (created_at, experiment_id, task_type, artifact_type, name,
                    procedure_code, verification_status, validation_score,
                    embedding_blob, version, meta_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    d.get("created_at") or now,
                    "migrated",
                    d["task_type"],
                    "harness",
                    name,
                    d.get("procedure_code", ""),
                    d.get("status", "active"),
                    score,
                    d.get("embedding"),
                    str(d.get("version", 1)),
                    json.dumps(meta, ensure_ascii=False),
                ),
            )
            n += 1
        except Exception as exc:
            print(f"  skip harness {name}: {exc}")
    return n


def migrate_records(leg: sqlite3.Connection, lts: sqlite3.Connection) -> int:
    """迁移 records。"""
    existing = {r[0] for r in lts.execute("SELECT task_type || '_' || COALESCE(experiment_id,'') FROM records")}
    rows = leg.execute("SELECT * FROM records").fetchall()
    cols = [d[0] for d in leg.execute("PRAGMA table_info(records)")]
    n = 0
    now = time.time()
    for row in rows:
        d = dict(zip(cols, row))
        # 计算 support_count
        try:
            src = json.loads(d.get("source_trajectories_json", "[]") or "[]")
            support = len(src) if isinstance(src, list) else 0
        except (json.JSONDecodeError, TypeError):
            support = 0
        dedup_key = f"{d['task_type']}_migrated"
        if dedup_key in existing:
            continue
        try:
            lts.execute(
                """INSERT INTO records
                   (created_at, experiment_id, task_type, preconditions_json,
                    param_steps_json, invariants_json, terminal_verifier,
                    source_trajectory_ids_json, support_count, version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    d.get("created_at") or now,
                    "migrated",
                    d["task_type"],
                    d.get("preconditions_json", "{}"),
                    d.get("canonical_steps_json", "[]"),
                    d.get("invariants_json", "[]"),
                    d.get("terminal_verifier", ""),
                    d.get("source_trajectories_json", "[]"),
                    support,
                    1,
                ),
            )
            existing.add(dedup_key)
            n += 1
        except Exception as exc:
            print(f"  skip record {d.get('id', '?')}: {exc}")
    return n


def migrate_stats(leg: sqlite3.Connection, lts: sqlite3.Connection) -> int:
    """迁移 stats（解析 stats_json 到独立列）。"""
    existing = {r[0] for r in lts.execute("SELECT task_type FROM stats")}
    rows = leg.execute("SELECT task_type, stats_json, updated_at FROM stats").fetchall()
    n = 0
    now = time.time()
    for task_type, stats_json, updated_at in rows:
        if task_type in existing:
            continue
        try:
            s = json.loads(stats_json or "{}")
        except json.JSONDecodeError:
            s = {}
        try:
            lts.execute(
                """INSERT INTO stats
                   (task_type, total_executions, agent_executions, harness_executions,
                    agent_successes, harness_successes, failure_counts_json,
                    estimated_token_savings, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_type,
                    s.get("total_executions", 0),
                    s.get("agent_executions", 0),
                    s.get("harness_executions", 0),
                    s.get("agent_successes", 0),
                    s.get("harness_successes", 0),
                    json.dumps(s.get("failure_counts", {}), ensure_ascii=False),
                    s.get("estimated_token_savings", 0),
                    updated_at or now,
                ),
            )
            n += 1
        except Exception as exc:
            print(f"  skip stats {task_type}: {exc}")
    return n


def main():
    if not LEGACY.exists():
        print(f"Legacy DB not found: {LEGACY}")
        return
    if not LTS.exists():
        print(f"LTS DB not found: {LTS}")
        return

    leg = sqlite3.connect(str(LEGACY))
    lts = sqlite3.connect(str(LTS))

    print("=== migrating embeddings ===")
    n = migrate_embeddings(leg, lts)
    print(f"  {n} new (skipped existing)")

    print("=== migrating ACTIVE harnesses → artifacts ===")
    n = migrate_harnesses(leg, lts)
    print(f"  {n} migrated")

    print("=== migrating records ===")
    n = migrate_records(leg, lts)
    print(f"  {n} migrated")

    print("=== migrating stats ===")
    n = migrate_stats(leg, lts)
    print(f"  {n} migrated")

    lts.commit()

    # verify
    print("\n=== LTS after migration ===")
    for t in ["trajectories", "substeps", "records", "artifacts", "stats", "embeddings"]:
        cnt = lts.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {cnt}")

    leg.close()
    lts.close()

    print("\n✅ Migration complete. Review above, then delete legacy DB manually:")
    print(f"   rm {LEGACY}")


if __name__ == "__main__":
    main()
