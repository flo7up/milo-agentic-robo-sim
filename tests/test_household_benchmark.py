import copy

import pytest

from scripts.benchmark_household import digest, planned_cases, score_attempt, suite_definition, summarize


def valid_metrics(**values):
    return {"recording_complete": True, "source_unchanged": True, "original_map_unchanged": True,
        "contact_episodes": 0, "manual_placements": 0, "elapsed_s": 5., "budget_s": 20., **values}


def test_household_suite_is_fixed_and_has_seven_tasks_three_repeats():
    definition = suite_definition()
    cases = planned_cases(definition)
    assert len(cases) == 21 and len({case["case_id"] for case in cases}) == 21
    assert {case["repeat"] for case in cases} == {1, 2, 3}
    assert digest(definition) == digest(suite_definition())
    assert definition["model_inference"] is False
    changed = copy.deepcopy(definition)
    changed["tasks"][0]["budget_s"] += 1
    assert digest(changed) != digest(definition)


def test_controller_completion_cannot_replace_independent_arrival():
    metrics = valid_metrics(arrivals=[{"controller_status": "completed", "position_error_m": .4, "stable_dwell_sim_s": 1.}])
    assert score_attempt("navigate", metrics)["status"] == "failed"
    metrics["arrivals"][0]["position_error_m"] = .1
    assert score_attempt("navigate", metrics)["passed"]
    assert not score_attempt("return_home", metrics)["passed"]
    metrics["recording_complete"] = False
    assert score_attempt("navigate", metrics)["status"] == "invalid"


def test_timeout_and_blocked_tasks_remain_in_denominators():
    definition = suite_definition()
    failed = score_attempt("localize", valid_metrics(localized=True, position_error_m=0., heading_error_rad=0., travel_m=0., elapsed_s=25.))
    assert failed["status"] == "failed"
    blocked = score_attempt("room_to_room", {"blocked_prerequisite": "Two named rooms required"})
    summary = summarize(definition, [{"case_id": "localize-r1", **failed}, {"case_id": "room_to_room-r1", **blocked}])
    assert summary["planned"] == 21 and summary["reported"] == 2 and summary["aggregate_success_rate"] is None
    assert summary["tasks"][0]["failed"] == 1 and summary["tasks"][0]["not_run"] == 2
    assert summary["tasks"][-1]["blocked"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(definition, [{"case_id": "localize-r1", **failed}] * 2)


def test_exploration_requires_useful_coverage_and_travel_not_only_a_timeout():
    metrics = valid_metrics(new_free_m2=1.6, travel_m=.226, buffers_empty=True)
    assert not score_attempt("explore", metrics)["passed"]
    metrics["travel_m"] = .8
    assert score_attempt("explore", metrics)["passed"]
    metrics["contact_episodes"] = 1
    assert not score_attempt("explore", metrics)["passed"]


def test_cancellation_requires_actual_motion_and_stale_rejection():
    metrics = valid_metrics(stop_triggered=True, travel_before_stop_m=.12, stop_ack_s=.05,
        frozen_after_stop=True, late_command_rejected=True, no_resume_motion=True)
    assert score_attempt("cancel", metrics)["passed"]
    metrics["travel_before_stop_m"] = 0.
    assert not score_attempt("cancel", metrics)["passed"]


def test_benchmark_map_preserves_geometry_and_original_document():
    from backend.home_mapping import HomeMap
    from scripts.benchmark_household import HOME_ID, benchmark_map
    home = HomeMap("shared_apartment_v1")
    home.evidence[180:220, 180:225] = -2
    home.places = [{"place_id": HOME_ID, "kind": "destination", "name": "Home", "pose_m_rad": [0., 0., 0.]}]
    document = home.document()
    before = digest(document)
    derived = benchmark_map(document)
    assert digest(document) == before
    assert derived["evidence"] == document["evidence"] and derived["visits"] == document["visits"]
    assert len(derived["places"]) == 2 and derived["places"][-1]["pose_m_rad"] == [.7, 0., 0.]


async def test_missing_room_prerequisite_is_recorded_without_starting_physics(tmp_path):
    from scripts.benchmark_household import execute_attempt
    case = next(case for case in planned_cases(suite_definition()) if case["task_id"] == "room_to_room")
    result = await execute_attempt(case, tmp_path / case["case_id"], {"places": []}, None, {}, tmp_path / "unused.sqlite3")
    assert result["status"] == "blocked" and not result["passed"]
    assert (tmp_path / case["case_id"] / "report.json").is_file()
    assert not (tmp_path / case["case_id"] / "recording").exists()


def test_comparison_rejects_changed_suite_or_source_drift(tmp_path):
    import json
    from scripts.benchmark_household import compare_runs
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    common = {"suite_sha256": "same", "derived_map_sha256": "map", "fixture_sha256": "fixture",
        "evidence": "scripted_test", "experiment_id": "first", "finished_at": "done", "source_changed_during_run": False}
    (baseline / "experiment.json").write_text(json.dumps(common))
    (candidate / "experiment.json").write_text(json.dumps({**common, "experiment_id": "second", "suite_sha256": "changed"}))
    with pytest.raises(ValueError, match="Unmatched"):
        compare_runs(baseline, candidate)
    (candidate / "experiment.json").write_text(json.dumps({**common, "experiment_id": "second", "source_changed_during_run": True}))
    with pytest.raises(ValueError, match="frozen"):
        compare_runs(baseline, candidate)


def test_archive_comparison_requires_complete_matching_frozen_inputs():
    from backend.saved_results import benchmark_comparability
    suite = suite_definition()
    manifest = {"suite": suite, "suite_sha256": digest(suite), "derived_map_sha256": "a" * 64,
        "fixture_sha256": "b" * 64, "design": {"source_sha256": "c" * 64, "runtime": {"renderer": "enhanced"}},
        "finished_at": "2026-09-15", "source_changed_during_run": False, "original_map_unchanged": True,
        "evidence": "scripted_test", "cases": planned_cases(suite)}
    trials = [{"benchmark": {"suite_id": suite["suite_id"], "task_id": case["task_id"], "status": "passed", "provenance_valid": True}} for case in manifest["cases"]]
    baseline = benchmark_comparability(manifest, trials)
    assert baseline["eligible"]
    candidate = copy.deepcopy(manifest)
    candidate["design"]["source_sha256"] = "d" * 64
    assert benchmark_comparability(candidate, trials)["cohort_id"] == baseline["cohort_id"]
    for field in ("derived_map_sha256", "fixture_sha256", "evidence", "reasoning"):
        altered = {**manifest, field: "e" * 64 if field.endswith("sha256") else "different"}
        assert benchmark_comparability(altered, trials)["cohort_id"] != baseline["cohort_id"]
    for field, value in (("finished_at", None), ("source_changed_during_run", True), ("original_map_unchanged", False),
            ("preflight_only", True), ("cases", manifest["cases"][:1])):
        assert not benchmark_comparability({**manifest, field: value}, trials)["eligible"]
    missing = copy.deepcopy(trials)
    missing[0]["benchmark"] = None
    assert not benchmark_comparability(manifest, missing)["eligible"]
    missing[0]["benchmark"] = {"suite_id": suite["suite_id"], "task_id": "localize", "status": "failed", "provenance_valid": False}
    assert not benchmark_comparability(manifest, missing)["eligible"]
    missing[0]["benchmark"] = {"suite_id": suite["suite_id"], "task_id": "localize", "status": "blocked", "provenance_valid": False}
    assert benchmark_comparability(manifest, missing)["eligible"]
    missing[0]["benchmark"]["task_id"] = "different"
    assert not benchmark_comparability(manifest, missing)["eligible"]


def test_benchmark_run_requires_explicit_baseline_or_new_series(tmp_path):
    from scripts.benchmark_household import argument_parser, validate_baseline
    options = argument_parser().parse_args(["--stage", "run", "--output", str(tmp_path / "new")])
    with pytest.raises(ValueError, match="Choose --baseline"):
        validate_baseline(options, suite_definition(), {}, [])
    options.establish_baseline = True
    assert validate_baseline(options, suite_definition(), {}, []) is None
    options.baseline = tmp_path
    with pytest.raises(ValueError, match="Choose --baseline"):
        validate_baseline(options, suite_definition(), {}, [])


def test_archive_keeps_benchmark_outcomes_separate_from_scenario_success(tmp_path):
    import json
    from backend.saved_results import saved_results
    definition = suite_definition()
    cases = planned_cases(definition)
    directory = tmp_path / "baseline"
    directory.mkdir()
    outcomes = [
        {**cases[0], "benchmark": {"suite_id": definition["suite_id"], "task_id": "localize", "status": "passed", "passed": True, "reason": "Measured pose"},
            "metrics": valid_metrics(), "recording_scorecard": {"complete_recording": True}, "physics_success": False},
        {**cases[6], "benchmark": {"suite_id": definition["suite_id"], "task_id": "room_to_room", "status": "blocked", "passed": False, "reason": "Rooms missing"}}]
    (directory / "experiment.json").write_text(json.dumps({"schema_version": 2, "stage": "spatial_workflow", "evidence": "scripted_test", "suite": definition, "cases": cases}))
    (directory / "summary.json").write_text(json.dumps(outcomes))
    batch = saved_results(tmp_path)["batches"][0]
    assert batch["trials"][0]["status"] == "Benchmark pass" and batch["successes"] == 0
    assert batch["trials"][6]["status"] == "Blocked prerequisite"
    assert batch["benchmark"]["tasks"][0]["passed"] == 1 and batch["benchmark"]["tasks"][0]["not_run"] == 2
    assert batch["benchmark"]["tasks"][-1]["blocked"] == 1
    outcomes[0]["metrics"]["source_unchanged"] = False
    (directory / "summary.json").write_text(json.dumps(outcomes))
    assert saved_results(tmp_path)["batches"][0]["trials"][0]["status"] == "Invalid benchmark"