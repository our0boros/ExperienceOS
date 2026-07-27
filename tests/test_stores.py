from experience_os.experience_library import (
    ExperienceLibrary,
    SubStepRecord,
    TrajectoryRecord,
)
from experience_os.stores import stores_for


def test_store_facades_write_and_query(tmp_path):
    library = ExperienceLibrary(tmp_path / "stores.db")
    traces, experiences, artifacts = stores_for(library)
    try:
        trace_id = traces.append(
            TrajectoryRecord(
                experiment_id="exp-1",
                method="react",
                domain="demo",
                task_id="task-1",
                task_type="lookup",
            )
        )
        assert trace_id == 1
        assert traces.query(experiment_id="exp-1")[0]["task_id"] == "task-1"

        record_id = experiences.log_record(
            task_type="lookup",
            preconditions={"email": "required"},
            param_steps=[{"template": "lookup({email})"}],
            invariants=["user exists"],
            source_ids=["task-1"],
            experiment_id="exp-1",
        )
        assert experiences.get_records("lookup")[0]["seq"] == record_id

        artifact_id = artifacts.log_artifact(
            task_type="lookup",
            artifact_type="harness",
            procedure_code="return lookup(email)",
            verification_status="verified",
            experiment_id="exp-1",
        )
        assert artifacts.get_artifacts(task_type="lookup")[0]["seq"] == artifact_id
    finally:
        library.close()


def test_consolidate_substeps_aggregates_and_deduplicates_evidence(tmp_path):
    library = ExperienceLibrary(tmp_path / "stores.db")
    traces, experiences, artifacts = stores_for(library)
    try:
        for index in range(3):
            task_id = f"task-{index}"
            traces.append(
                TrajectoryRecord(
                    experiment_id="exp-1", method="react", domain="demo",
                    task_id=task_id, task_type="lookup", success=True,
                )
            )
            experiences.library.log_substep(
                SubStepRecord(
                    trajectory_id=task_id, experiment_id="exp-1", plan_idx=0,
                    intent="lookup_user", tool_name="find_user", success=True,
                    parent_task_type="lookup", parent_task_success=True,
                )
            )
        # Repeated observation in one trajectory must not inflate support.
        experiences.library.log_substep(
            SubStepRecord(
                trajectory_id="task-0", experiment_id="exp-1", plan_idx=1,
                intent="lookup_user", tool_name="find_user", success=True,
                parent_task_type="lookup", parent_task_success=True,
            )
        )

        candidates = experiences.consolidate_substeps(
            experiment_id="exp-1", min_support=3, max_candidates=10,
        )
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["support_count"] == 3
        assert candidate["evidence"]["trajectory_ids"] == ["task-0", "task-1", "task-2"]
        assert candidate["source"] == "substeps"
        assert candidate["reason"]
        assert experiences.candidate_stats()["dedup_count"] == 1
        assert experiences.candidate_stats()["accepted_count"] == 1
        assert artifacts.get_artifacts(status="") == []
    finally:
        library.close()


def test_consolidate_substeps_filters_low_support(tmp_path):
    library = ExperienceLibrary(tmp_path / "stores.db")
    _, experiences, _ = stores_for(library)
    try:
        for index in range(2):
            experiences.library.log_substep(
                SubStepRecord(
                    trajectory_id=f"task-{index}", experiment_id="exp-1", plan_idx=0,
                    intent="rare_lookup", tool_name="find_user", success=True,
                    parent_task_type="lookup", parent_task_success=True,
                )
            )
        assert experiences.consolidate_substeps(
            experiment_id="exp-1", min_support=3,
        ) == []
        stats = experiences.candidate_stats()
        assert stats["candidate_count"] == 1
        assert stats["accepted_count"] == 0
        assert stats["filtered_count"] == 1
    finally:
        library.close()


def test_raw_trace_does_not_automatically_create_artifact(tmp_path):
    library = ExperienceLibrary(tmp_path / "stores.db")
    traces, _, artifacts = stores_for(library)
    try:
        traces.append(
            TrajectoryRecord(
                experiment_id="exp-1",
                method="react",
                domain="demo",
                task_id="task-1",
                task_type="lookup",
            )
        )
        assert artifacts.get_artifacts(task_type="lookup", status="")[0:] == []
    finally:
        library.close()
