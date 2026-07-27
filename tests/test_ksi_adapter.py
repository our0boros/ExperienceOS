from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experience_os.ksi_adapter import (
    KsiRunSpec,
    KsiTaskSpec,
    build_ksi_run_spec,
    build_ksi_task_spec,
    build_ksi_task_specs,
    export_ksi_run_manifest,
    export_ksi_tasks,
)


# ── τ-bench task fixture ──────────────────────────────────────────────
def _make_task(task_id="42", reason="return an item", known="email: a@b.com",
               actions=None, ticket=None, purpose=None):
    """Build a τ-bench-shaped task object."""
    instructions = SimpleNamespace(
        reason_for_call=reason,
        known_info=known,
        unknown_info="unknown details",
        task_instructions="Please help the customer",
    )
    if actions is None:
        actions = [
            SimpleNamespace(name="get_order_details",
                            arguments={"order_id": "#ORD99"},
                            requestor="assistant"),
            SimpleNamespace(name="exchange_delivered_order_items",
                            arguments={"item_ids": ["item_1"]},
                            requestor="assistant"),
        ]
    return SimpleNamespace(
        id=task_id,
        task_id=task_id,
        user_scenario=SimpleNamespace(instructions=instructions),
        evaluation_criteria=SimpleNamespace(
            actions=actions,
            common_condition="DB hash matches",
        ),
        initial_state=SimpleNamespace(
            initialization_data=None,
            initialization_actions=[],
            message_history=[],
        ),
        description=SimpleNamespace(purpose=purpose or ""),
        ticket=ticket or "",
    )


# ── build_ksi_task_spec ───────────────────────────────────────────────
class TestBuildKsiTaskSpec:
    def test_extracts_prompt_from_instructions(self):
        task = _make_task(task_id="1", reason="customer wants a refund")
        spec = build_ksi_task_spec(task)
        assert "customer wants a refund" in spec.prompt
        assert "a@b.com" in spec.prompt
        assert "Please help the customer" in spec.prompt
        assert spec.task_id == "tau2-1"

    def test_includes_task_type_in_metadata(self):
        task = _make_task(actions=[
            SimpleNamespace(name="cancel_pending_order",
                            arguments={"order_id": "#X"}, requestor="assistant"),
        ])
        spec = build_ksi_task_spec(task)
        assert spec.metadata["task_type"] == "cancel_pending_order"
        assert spec.metadata["task_source"] == "tau2-bench"
        assert spec.metadata["num_reference_actions"] == 1

    def test_no_eval_field_for_tau2_tasks(self):
        spec = build_ksi_task_spec(_make_task())
        assert spec.eval is None

    def test_handles_minimal_task(self):
        task = SimpleNamespace(id="99")
        spec = build_ksi_task_spec(task)
        assert spec.task_id == "tau2-99"
        assert "tau2 task 99" in spec.prompt
        assert spec.metadata["task_type"] == "unknown"

    def test_respects_explicit_task_type(self):
        spec = build_ksi_task_spec(_make_task(), task_type="custom_type")
        assert spec.metadata["task_type"] == "custom_type"

    def test_ksi_dict_excludes_internal_metadata(self):
        spec = build_ksi_task_spec(_make_task())
        d = spec.to_ksi_dict()
        assert "task_id" in d
        assert "prompt" in d
        assert "task_source" not in d  # internal only
        assert "num_reference_actions" not in d


# ── build_ksi_task_specs ──────────────────────────────────────────────
def test_build_ksi_task_specs_batch():
    tasks = [_make_task(task_id=str(i)) for i in range(3)]
    specs = build_ksi_task_specs(tasks)
    assert len(specs) == 3
    assert specs[0].task_id == "tau2-0"
    assert specs[2].task_id == "tau2-2"


# ── build_ksi_run_spec ────────────────────────────────────────────────
def test_build_ksi_run_spec_collects_task_types():
    tasks = [
        _make_task(task_id="1", actions=[
            SimpleNamespace(name="get_order_details",
                            arguments={}, requestor="assistant"),
        ]),
        _make_task(task_id="2", actions=[
            SimpleNamespace(name="cancel_pending_order",
                            arguments={}, requestor="assistant"),
        ]),
    ]
    spec = build_ksi_run_spec(tasks, domain="retail",
                              model="deepseek-v4", model_provider="deepinfra",
                              warmup=5, eval_size=10, split_policy="train_test",
                              experiment_id="exp-ksi-1")
    assert spec.domain == "retail"
    assert spec.model == "deepseek-v4"
    assert spec.model_provider == "deepinfra"
    assert spec.warmup_count == 5
    assert spec.eval_count == 10
    assert spec.split_policy == "train_test"
    assert spec.experiment_id == "exp-ksi-1"
    assert "cancel_pending_order" in spec.task_types
    assert "get_order_details" in spec.task_types
    assert "tau2-bench" in spec.evaluator


# ── export_ksi_tasks ────────────────────────────────────────────────
def test_export_ksi_tasks_writes_valid_jsonl(tmp_path):
    tasks = [_make_task(task_id=str(i)) for i in range(2)]
    output = str(tmp_path / "tasks.jsonl")
    result = export_ksi_tasks(tasks, output)
    assert result == str(Path(output).resolve())
    assert Path(output).exists()

    lines = Path(output).read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "task_id" in record
        assert "prompt" in record


def test_export_ksi_tasks_creates_parent_dirs(tmp_path):
    tasks = [_make_task()]
    output = str(tmp_path / "sub" / "dir" / "tasks.jsonl")
    export_ksi_tasks(tasks, output)
    assert Path(output).exists()


# ── export_ksi_run_manifest ──────────────────────────────────────────
def test_export_ksi_run_manifest_writes_valid_json(tmp_path):
    tasks = [_make_task()]
    output = str(tmp_path / "manifest.json")
    result = export_ksi_run_manifest(tasks, output, domain="airline")
    assert result == str(Path(output).resolve())
    assert Path(output).exists()

    data = json.loads(Path(output).read_text(encoding="utf-8"))
    assert data["domain"] == "airline"
    assert data["task_count"] == 1
    assert data["evaluator"] == "tau2-bench"


# ── KsiTaskSpec dataclass ─────────────────────────────────────────────
def test_ksi_task_spec_defaults():
    spec = KsiTaskSpec(task_id="t1", prompt="do something")
    d = spec.to_ksi_dict()
    assert d == {"task_id": "t1", "prompt": "do something"}
    # no workspace_dir, files, or eval by default


def test_ksi_task_spec_with_eval():
    spec = KsiTaskSpec(task_id="t2", prompt="test",
                       eval={"command": "pytest", "timeout_sec": 60})
    d = spec.to_ksi_dict()
    assert d["eval"] == {"command": "pytest", "timeout_sec": 60}


def test_ksi_task_spec_with_workspace():
    spec = KsiTaskSpec(task_id="t3", prompt="code",
                       workspace_dir="/tmp/ws",
                       files={"main.py": "print(1)"})
    d = spec.to_ksi_dict()
    assert d["workspace_dir"] == "/tmp/ws"
    assert d["files"] == {"main.py": "print(1)"}


# ── KsiRunSpec dataclass ─────────────────────────────────────────────
def test_ksi_run_spec_defaults():
    spec = KsiRunSpec()
    assert spec.task_source == "custom"
    assert spec.evaluator == "command"
    assert spec.max_steps == 30
