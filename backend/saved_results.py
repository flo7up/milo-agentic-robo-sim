import hashlib
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from backend.recording import read_recording_json


RESULTS_ROOT = Path(__file__).resolve().parents[1] / ".runtime"
MAX_FILE_BYTES = 8_000_000
MAX_TRAJECTORY_BYTES = 64_000_000


def local_file(path, root):
    resolved = path.resolve()
    return resolved.is_relative_to(root.resolve()) and resolved == path.absolute() and path.is_file()


def read_json(path, root):
    if not local_file(path, root) or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Unavailable saved report")
    return read_recording_json(path)


def text(value, fallback="Unknown"):
    return value[:240] if isinstance(value, str) and value else fallback


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def result_directories(root):
    paths = set(root.glob("*/experiment.json")) | set(root.glob("*/*/experiment.json"))
    index = root / "performance/index.jsonl"
    if local_file(index, root) and index.stat().st_size <= MAX_FILE_BYTES:
        for line in index.read_text(encoding="utf-8").splitlines()[-2000:]:
            try:
                directory = Path(json.loads(line)["output"])
                candidate = directory / "experiment.json"
                if local_file(candidate, root):
                    paths.add(candidate)
            except (ValueError, KeyError, TypeError, OSError):
                continue
    return sorted((path.parent for path in paths if local_file(path, root)),
        key=lambda directory: (directory / "experiment.json").stat().st_mtime, reverse=True)


def normalize_trial(result, case, index, batch_id, directory, root, evidence):
    result = result if isinstance(result, dict) else {}
    score = result.get("recording_scorecard") or {}
    score = score if isinstance(score, dict) else {}
    challenge = result.get("challenge_status") or {}
    challenge = challenge if isinstance(challenge, dict) else {}
    physics = result.get("physics_success")
    assisted = bool(result.get("manual_placements") or score.get("operator_assisted"))
    complete = score.get("complete_recording") is True
    last_updated = number(result.get("updated_at"))
    unfinished = result.get("run_status") == "running"
    running = unfinished and last_updated is not None and 0 <= time.time() - last_updated <= 15.
    verified = (physics is True and complete and not assisted and result.get("test_passed") is not False
        and not unfinished and result.get("verification_eligible") is not False and number(score.get("completion_time_s")) is not None)
    status = "Verified pass" if verified else "Missing report" if not result else "Incomplete recording" if not complete else "Not passed"
    if physics is True and not verified:
        status = "Assisted pass" if assisted else "Unverified pass"
    if result.get("test_passed") is False:
        status = "Test failed"
    if unfinished:
        status = "Running" if running else "Interrupted recording"
    if result.get("verification_eligible") is False and result.get("evidence") == "operator_session" and complete and not unfinished:
        status = "Recorded workflow"
    case_id = text(case.get("case_id"), text(result.get("case_id"), "case"))
    terminal = directory / case_id / "terminal.png"
    safe_case = case_id not in {".", ".."} and "/" not in case_id and "\\" not in case_id
    has_image = safe_case and local_file(terminal, root)
    has_route = safe_case and local_file(directory / case_id / "recording/trajectory.jsonl", root)
    spatial = score.get("spatial_recording") or {}
    spatial = spatial if isinstance(spatial, dict) else {}
    benchmark = result.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, dict) else None
    if benchmark:
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        outcome = benchmark.get("status")
        valid = complete and all(metrics.get(key) is True for key in ("source_unchanged", "original_map_unchanged", "recording_complete"))
        outcome = "invalid" if outcome == "passed" and (not valid or benchmark.get("passed") is not True) else outcome
        status = {"passed": "Benchmark pass", "failed": "Benchmark failed", "blocked": "Blocked prerequisite", "invalid": "Invalid benchmark"}.get(outcome, "Unscored benchmark")
        benchmark = {"suite_id": text(benchmark.get("suite_id")), "task_id": text(benchmark.get("task_id")),
            "provenance_valid": valid and not assisted,
            "status": outcome, "reason": text(benchmark.get("reason")), "criteria": text(result.get("criteria")),
            "budget_s": number(result.get("budget_s")), "setup_s": number(metrics.get("setup_s")),
            "position_error_m": number(metrics.get("position_error_m")), "new_free_m2": number(metrics.get("new_free_m2"))}
        verified = False
    return {"case_id": case_id, "challenge": text(case.get("challenge"), text(result.get("challenge"))),
        "benchmark": benchmark,
        "challenge_sha256": text(case.get("challenge_sha256"), ""), "running": running, "updated_at": last_updated,
        "task_sha256": hashlib.sha256(json.dumps([case.get("challenge_sha256"), result.get("initial_goal"), result.get("final_goal")], sort_keys=True).encode()).hexdigest(),
        "goal": text(result.get("final_goal"), text(challenge.get("goal"), "Scenario goal")),
        "environment": text(case.get("environment"), text(challenge.get("environment"), "standalone")),
        "title": text(result.get("title"), text(case.get("title"))) if benchmark else text(challenge.get("title"), text(result.get("challenge"), text(case.get("challenge")))),
        "evidence": text(result.get("evidence"), evidence), "status": status, "verified_success": verified,
        "physics_success": physics if isinstance(physics, bool) else None, "recording_complete": complete, "assisted": assisted,
        "completion_s": number(score.get("completion_time_s")) if verified else None,
        "elapsed_s": number(result.get("evaluation_elapsed_s", result.get("wall_s"))),
        "distance_m": number(score.get("actual_distance_m")), "contact_episodes": number(score.get("contact_episodes")),
        "mean_translation_speed_mps": number(score.get("mean_translation_speed_mps")),
        "stationary_wall_s": number(score.get("stationary_wall_s")), "route_arrivals": number(score.get("route_arrivals")),
        "input_tokens": number(result.get("input_tokens")), "output_tokens": number(result.get("output_tokens")),
        "inference_median_s": number(result.get("inference_median_s")), "turns": number(result.get("supervisor_turns")),
        "termination": text(result.get("termination_reason"), text(result.get("phase"))), "rendering": text(result.get("rendering")),
        "false_completion_claim": result.get("unverified_completion_claim") is True,
        "trajectory_url": f"/api/test-results/{batch_id}/trajectories/{index}" if has_route else None,
        "replay_url": f"/api/test-results/{batch_id}/replay/{index}" if has_route else None,
        "spatial_progress": {"snapshots": number(spatial.get("snapshots")), "events": number(spatial.get("events")),
            "map_id": text(spatial.get("map_id"), ""), "map_revision": number(spatial.get("map_revision")),
            "complete": spatial.get("complete") is True, "initial": spatial.get("initial"), "final": spatial.get("final"),
            "final_task": spatial.get("final_task")},
        "image_url": f"/api/test-results/{batch_id}/images/{index}" if has_image else None}


def normalized_variant(manifest):
    architecture = manifest.get("architecture")
    model = manifest.get("model_variant")
    if not isinstance(architecture, dict) or not isinstance(model, dict):
        return {"architecture": None, "model_variant": None, "variant_id": None}
    configuration = model.get("configuration")
    configuration = configuration if isinstance(configuration, dict) else {}
    return {"architecture": {key: text(architecture.get(key), "Unknown")
                for key in ("id", "name", "version", "revision", "version_key", "implementation_sha256")},
        "model_variant": {**{key: text(model.get(key), "Unknown") for key in ("provider", "profile_id", "deployment", "revision")},
            "underlying_identity_verified": model.get("underlying_identity_verified") is True,
            "configuration": {key: value for key, value in configuration.items()
                if key in {"reasoning", "images_per_request", "context_tokens", "feedback_interval_s", "max_turns",
                    "compact_arms", "continuous_handoff", "adaptive_navigation", "ai_generated_routes", "skill_composer", "navigation_backend",
                    "session_timeout_s", "camera_history", "local_checkpoint"}
                and (value is None or isinstance(value, bool) or number(value) is not None or isinstance(value, str) and len(value) <= 80)}},
        "variant_id": text(manifest.get("variant_id"), "Unknown")}


def benchmark_comparability(manifest, trials):
    suite = manifest.get("suite")
    if not isinstance(suite, dict) or not isinstance(suite.get("tasks"), list):
        return None
    reasons = []
    fingerprint = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    for field in ("suite_sha256", "derived_map_sha256", "fixture_sha256"):
        if not isinstance(manifest.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[field]):
            reasons.append("Missing " + field.replace("_sha256", "").replace("_", " ") + " fingerprint")
    if manifest.get("suite_sha256") != fingerprint(suite):
        reasons.append("Suite fingerprint does not match its definition")
    design = manifest.get("design") if isinstance(manifest.get("design"), dict) else {}
    if not design.get("source_sha256") or not design.get("runtime"):
        reasons.append("Source or runtime provenance missing")
    if not manifest.get("finished_at"):
        reasons.append("Run is not finalized")
    if manifest.get("source_changed_during_run") is not False:
        reasons.append("Source stability not established")
    if manifest.get("original_map_unchanged") is not True:
        reasons.append("Map stability not established")
    if manifest.get("preflight_only"):
        reasons.append("Preflight only")
    if manifest.get("evidence") not in {"scripted_test", "real_model", "scripted_reference"}:
        reasons.append("Evidence is not a controlled test series")
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases) or not all(isinstance(task, dict) for task in suite["tasks"]):
        return {"eligible": False, "reasons": ["Malformed benchmark definition"], "cohort_id": "", "experiment_id": "", "baseline_experiment_id": ""}
    task_ids = [task.get("id") for task in suite["tasks"]]
    repeats = len(suite.get("start_offsets_m", [])) or suite.get("repetitions", 0)
    if (not repeats or len(cases) != repeats * len(task_ids)
            or len({case.get("case_id") for case in cases}) != len(cases)
            or any(sum(case.get("task_id") == task_id for case in cases) != repeats for task_id in task_ids)):
        reasons.append("Planned cases do not match the full suite")
    for index, trial in enumerate(trials):
        result = trial.get("benchmark") or {}
        if result.get("status") not in {"passed", "failed", "blocked"}:
            reasons.append("Missing, invalid or unfinished outcomes")
        elif result["status"] != "blocked" and not result.get("provenance_valid"):
            reasons.append("Incomplete or assisted trial evidence")
        if result.get("suite_id") != suite.get("suite_id"):
            reasons.append("Trial belongs to a different suite")
        if index >= len(cases) or result.get("task_id") != cases[index].get("task_id"):
            reasons.append("Trial task does not match the planned case")
    settings = {field: manifest.get(field) for field in ("mode", "evidence", "reasoning", "supervisor_deployment",
        "images_per_request", "context_tokens", "feedback_interval_s", "session_timeout_s", "camera_history",
        "continuous_handoff", "skill_composer", "exploration")}
    if manifest.get("evidence") == "real_model" and not isinstance(manifest.get("model_variant"), dict):
        reasons.append("Model configuration provenance missing")
    inputs = {field: manifest.get(field) for field in ("suite_sha256", "derived_map_sha256", "fixture_sha256")}
    model = manifest.get("model_variant") if isinstance(manifest.get("model_variant"), dict) else {}
    inputs.update(settings=settings, runtime=design.get("runtime"),
        model_configuration=model.get("configuration"),
        cases=[{key: case.get(key) for key in ("case_id", "task_id", "challenge_sha256", "budget_s", "start_offset_m")} for case in cases])
    return {"eligible": not reasons, "reasons": list(dict.fromkeys(reasons)),
        "cohort_id": fingerprint(inputs), "experiment_id": text(manifest.get("experiment_id"), ""),
        "baseline_experiment_id": text(manifest.get("baseline_experiment_id"), "")}


def saved_results(root=None):
    root = root or RESULTS_ROOT
    batches, skipped = [], 0
    directories = result_directories(root)
    for directory in directories[:500]:
        try:
            manifest = read_json(directory / "experiment.json", root)
            if not isinstance(manifest, dict) or manifest.get("stage", "challenges") not in {"challenges", "spatial_workflow"}:
                continue
            cases = manifest.get("cases")
            summary = directory / "summary.json"
            results = read_json(summary, root) if summary.exists() else []
            if isinstance(results, dict):
                results = [results]
            if not isinstance(results, list):
                raise ValueError("Invalid result list")
            if cases is None:
                cases = [{"case_id": result.get("case_id", result.get("challenge")), "challenge": result.get("challenge")}
                         for result in results if isinstance(result, dict) and isinstance(result.get("challenge"), str)]
                declared = manifest.get("challenges", [])
                if isinstance(declared, list):
                    cases.extend({"case_id": name, "challenge": name} for name in declared if isinstance(name, str)
                                 and not any(case["challenge"] == name for case in cases))
            if not isinstance(cases, list) or not cases or not all(isinstance(case, dict) for case in cases):
                continue
            by_case = {result.get("case_id", result.get("challenge")): result for result in results if isinstance(result, dict)}
            identifier = hashlib.sha256(directory.relative_to(root).as_posix().encode()).hexdigest()[:20]
            design = manifest.get("design") if isinstance(manifest.get("design"), dict) else {}
            evidence = text(manifest.get("evidence"), "scripted_reference" if manifest.get("mode") == "reference" else
                            "real_model" if results and all(isinstance(result, dict) and result.get("real_luna") is True for result in results) else "unknown")
            trials = []
            for index, case in enumerate(cases[:500]):
                case_id = case.get("case_id")
                result = by_case.get(case_id)
                if result is None and isinstance(case_id, str) and case_id not in {".", ".."} and "/" not in case_id and "\\" not in case_id:
                    report = directory / case_id / "report.json"
                    if report.exists():
                        result = read_json(report, root)
                trials.append(normalize_trial(result, case, index, identifier, directory, root, evidence))
            modified = (directory / "experiment.json").stat().st_mtime
            suite = manifest.get("suite") if isinstance(manifest.get("suite"), dict) else None
            benchmark_summary = None
            if suite and isinstance(suite.get("tasks"), list):
                task_rows = []
                for task in suite["tasks"][:30]:
                    if not isinstance(task, dict):
                        continue
                    selected_cases = [index for index, case in enumerate(cases[:500]) if case.get("task_id") == task.get("id")]
                    outcomes = [(trials[index].get("benchmark") or {}).get("status", "not_run") for index in selected_cases]
                    times = [trials[index]["elapsed_s"] for index in selected_cases
                        if (trials[index].get("benchmark") or {}).get("status") == "passed" and trials[index]["elapsed_s"] is not None]
                    task_rows.append({"task_id": text(task.get("id")), "title": text(task.get("title")), "planned": len(selected_cases),
                        "successful_median_s": statistics.median(times) if times else None,
                        **{key: outcomes.count(key) for key in ("passed", "failed", "blocked", "invalid", "not_run")}})
                benchmark_summary = {"suite_id": text(suite.get("suite_id")), "suite_sha256": text(manifest.get("suite_sha256")),
                    "map_sha256": text(manifest.get("derived_map_sha256")), "preflight": manifest.get("preflight_only") is True,
                    "comparison": benchmark_comparability(manifest, trials),
                    "tasks": task_rows}
            batches.append({"id": identifier, "name": directory.relative_to(root).as_posix(),
                "benchmark": benchmark_summary,
                "home_map": {key: (manifest.get("home_map") or {}).get(key) for key in ("mode", "map_id", "name", "revision", "sha256", "localization")}
                    if isinstance(manifest.get("home_map"), dict) else None,
                "session_id": text(manifest.get("session_id"), ""), "running": any(trial["running"] for trial in trials),
                **normalized_variant(manifest),
                "date": text(manifest.get("started_at"), datetime.fromtimestamp(modified, timezone.utc).isoformat()),
                "date_source": "recorded" if manifest.get("started_at") else "file_modified",
                "design": text(design.get("label"), "Legacy / unlabelled"), "source_sha256": text(design.get("source_sha256"), ""),
                "mode": text(manifest.get("mode")), "evidence": evidence, "model": text(manifest.get("supervisor_deployment")),
                "reasoning": text(manifest.get("reasoning")), "budget_s": number(manifest.get("session_timeout_s")),
                "history": text(manifest.get("camera_history"), "Not recorded"), "legacy": manifest.get("schema_version") != 2,
                "source_changed": manifest.get("source_changed_during_run") is True,
                "planned": len(cases), "successes": sum(trial["verified_success"] for trial in trials), "trials": trials})
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            skipped += 1
    return {"batches": sorted(batches, key=lambda batch: batch["date"], reverse=True), "skipped": skipped,
            "truncated": len(directories) > 500,
            "scope": "Model/robot sessions and evaluations, with separately labelled scripted evidence; in-progress recordings are provisional."}


def scene_vector(value, length, limit=1000):
    if (not isinstance(value, list) or len(value) != length
            or not all(isinstance(item, (int, float)) and not isinstance(item, bool)
                       and math.isfinite(item) and abs(item) <= limit for item in value)):
        raise ValueError("Invalid scene vector")
    return value


def normalize_scene(data, source):
    if data.get("schema_version") != 1 or data.get("coordinate_frame") != "recorded_world_xy_m":
        raise ValueError("Unknown scene format")
    geometry, snapshot = data["geometry"], data["snapshot"]
    poses = snapshot["poses"]
    if not isinstance(geometry, list) or not isinstance(poses, list) or max(len(geometry), len(poses)) > 2000:
        raise ValueError("Scene exceeds display limits")
    transforms = {}
    for pose in poses:
        key = pose["key"]
        if not isinstance(key, str) or not re.fullmatch(r"[0-9]+:-?[0-9]+", key) or key in transforms:
            raise ValueError("Invalid scene pose key")
        quaternion = scene_vector(pose["quaternion"], 4, 1)
        if not .99 <= sum(value * value for value in quaternion) <= 1.01:
            raise ValueError("Invalid scene rotation")
        transforms[key] = {"key": key, "position": scene_vector(pose["position"], 3), "quaternion": quaternion}
    assets = []
    for asset in geometry:
        key = asset["key"]
        if not isinstance(key, str) or key not in transforms:
            raise ValueError("Missing scene transform")
        if key.startswith(f"{data['robot_body_id']}:"):
            continue
        shape = asset["type"]
        dimensions = scene_vector(asset["dimensions"], 3)
        if shape not in {2, 3, 4} or isinstance(shape, bool) or min(dimensions) < 0 or dimensions[0] <= 0:
            raise ValueError("Invalid scene shape")
        if shape in {3, 4} and dimensions[1] <= 0 or shape == 3 and dimensions[2] <= 0:
            raise ValueError("Invalid scene dimensions")
        color = scene_vector(asset["color"], 4, 1)
        if min(color) < 0:
            raise ValueError("Invalid scene color")
        quaternion = scene_vector(asset["quaternion"], 4, 1)
        if not .99 <= sum(value * value for value in quaternion) <= 1.01:
            raise ValueError("Invalid scene rotation")
        texture = asset.get("texture")
        assets.append({"key": key, "type": shape, "dimensions": dimensions, "color": color,
            "name": text(asset.get("name"), "Object"), "position": scene_vector(asset["position"], 3),
            "quaternion": quaternion, "texture": texture if isinstance(texture, str)
            and re.fullmatch(r"/api/textures/[a-z]+\.png", texture) else None})
    if not assets:
        raise ValueError("Empty scene")
    return {"source": source, "geometry": assets, "poses": [transforms[key] for key in dict.fromkeys(asset["key"] for asset in assets)],
        "simulated_s": number(snapshot.get("simulated_time_s")), "evaluation_only": True}


def trajectory_scene(directory, challenge_id, root, environment="standalone"):
    path = directory / "scene.json"
    try:
        if path.exists() or path.is_symlink():
            return normalize_scene(read_json(path, root), "recorded_initial")
        from backend.challenges import get_challenge, shared_apartment
        from backend.materials import scene_material
        if environment not in {"standalone", "shared_apartment_v1"}:
            return None
        challenge = shared_apartment(challenge_id) if environment == "shared_apartment_v1" else get_challenge(challenge_id)
        if challenge is None:
            return None
        geometry, poses = [], []
        for index, item in enumerate(challenge.scene()):
            key = f"{index}:-1"
            cylinder = item.get("shape") == "cylinder"
            size = item["size"]
            texture = item.get("texture") or scene_material(item, challenge_id in {"apartment", "kitchen_bathroom", "clinic_delivery"})
            geometry.append({"key": key, "name": item["name"], "type": 4 if cylinder else 3,
                "dimensions": [size[2], size[0] / 2, 0] if cylinder else list(size), "color": list(item["color"]),
                "position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "texture": f"/api/textures/{texture}.png" if texture else None})
            poses.append({"key": key, "position": list(item["position"]), "quaternion": [0, 0, 0, 1]})
        return normalize_scene({"schema_version": 1, "coordinate_frame": "recorded_world_xy_m", "robot_body_id": -1,
            "geometry": geometry, "snapshot": {"poses": poses, "simulated_time_s": 0}}, "reconstructed_current")
    except (OSError, ValueError, TypeError, KeyError, OverflowError, AttributeError):
        return None


def recorded_trajectory(batch_id, trial_index, root=None):
    root = root or RESULTS_ROOT
    batch = next((batch for batch in saved_results(root)["batches"] if batch["id"] == batch_id), None)
    if batch is None or not 0 <= trial_index < len(batch["trials"]) or not batch["trials"][trial_index]["trajectory_url"]:
        raise ValueError("Saved trajectory unavailable")
    path = root / batch["name"] / batch["trials"][trial_index]["case_id"] / "recording/trajectory.jsonl"
    if not local_file(path, root) or path.stat().st_size > MAX_TRAJECTORY_BYTES:
        raise ValueError("Saved trajectory unavailable")
    points, contacts = [], []
    previous_contact, previous_placement, segment = False, None, 0
    bounds = [math.inf, math.inf, -math.inf, -math.inf]
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 65536 or len(points) >= 50000:
                raise ValueError("Saved trajectory exceeds display limits")
            sample = json.loads(line)
            if not isinstance(sample, dict):
                raise ValueError("Invalid trajectory sample")
            position, elapsed = sample.get("position_m"), number(sample.get("wall_s"))
            if (not isinstance(position, list) or len(position) < 2 or elapsed is None
                    or (points and elapsed < points[-1]["wall_s"])
                    or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                               and math.isfinite(value) and abs(value) <= 1000 for value in position[:2])):
                raise ValueError("Invalid trajectory coordinates or time")
            placement = sample.get("manual_placements", 0)
            if points and placement != previous_placement:
                segment += 1
            previous_placement = placement
            point = {"x": position[0], "y": position[1], "wall_s": elapsed, "segment": segment,
                "status": text(sample.get("physics_status")), "activity": text(sample.get("activity"))}
            points.append(point)
            contact = bool(sample.get("collisions"))
            if contact and not previous_contact:
                contacts.append(point)
            previous_contact = contact
            bounds = [min(bounds[0], point["x"]), min(bounds[1], point["y"]),
                      max(bounds[2], point["x"]), max(bounds[3], point["y"])]
    if not points:
        raise ValueError("Saved trajectory is empty")
    stride = max(1, math.ceil(len(points) / 1500))
    selected = sorted(set(range(0, len(points), stride)) | {len(points) - 1})
    return {"coordinate_frame": "recorded_world_xy_m", "sample_count": len(points), "bounds_m": bounds,
        "downsampled": stride > 1, "points": [points[index] for index in selected],
        "contacts": contacts[:200], "contact_markers_truncated": len(contacts) > 200,
        "scene": trajectory_scene(path.parent, batch["trials"][trial_index]["challenge"], root, batch["trials"][trial_index]["environment"]),
        "evaluation_only": True}


def terminal_image(batch_id, trial_index, root=None):
    root = root or RESULTS_ROOT
    catalog = saved_results(root)
    batch = next((batch for batch in catalog["batches"] if batch["id"] == batch_id), None)
    if batch is None or not 0 <= trial_index < len(batch["trials"]) or not batch["trials"][trial_index]["image_url"]:
        raise ValueError("Saved camera image unavailable")
    path = root / batch["name"] / batch["trials"][trial_index]["case_id"] / "terminal.png"
    if not local_file(path, root) or path.stat().st_size > MAX_FILE_BYTES or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Saved camera image unavailable")
    return path


@lru_cache(maxsize=128)
def replay_batch_directory(batch_id, root):
    directory = next((directory for directory in result_directories(root)
        if hashlib.sha256(directory.relative_to(root).as_posix().encode()).hexdigest()[:20] == batch_id), None)
    if directory is None:
        raise ValueError("Unknown recorded trial")
    return directory


def replay_directory(batch_id, trial_index, root):
    if not re.fullmatch(r"[a-f0-9]{20}", batch_id):
        raise ValueError("Unknown recorded trial")
    batch_directory = replay_batch_directory(batch_id, root)
    manifest = read_json(batch_directory / "experiment.json", root)
    cases = manifest.get("cases")
    summary = batch_directory / "summary.json"
    results = read_json(summary, root) if summary.exists() else []
    if isinstance(results, dict):
        results = [results]
    if cases is None:
        cases = [{"case_id": result.get("case_id", result.get("challenge")), "challenge": result.get("challenge")}
            for result in results if isinstance(result, dict) and isinstance(result.get("challenge"), str)]
    if not isinstance(cases, list) or not 0 <= trial_index < min(len(cases), 500):
        raise ValueError("Unknown recorded trial")
    case = cases[trial_index]
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or case_id in {".", ".."} or "/" in case_id or "\\" in case_id:
        raise ValueError("Invalid recording case")
    report = batch_directory / case_id / "report.json"
    result = read_json(report, root) if report.exists() else next((result for result in results
        if isinstance(result, dict) and result.get("case_id", result.get("challenge")) == case_id), {})
    trial = normalize_trial(result, case, trial_index, batch_id, batch_directory, root, text(manifest.get("evidence")))
    directory = batch_directory / case_id / "recording"
    path = directory / "trajectory.jsonl"
    if not local_file(path, root) or path.stat().st_size > MAX_TRAJECTORY_BYTES:
        raise ValueError("Recorded replay unavailable")
    return directory, path, trial


def replay_samples(path):
    previous = -1.
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= 50000 or len(line) > 262144:
                raise ValueError("Replay exceeds display limits")
            sample = json.loads(line)
            elapsed = number(sample.get("wall_s"))
            if elapsed is None or elapsed < previous:
                raise ValueError("Replay timestamps are invalid")
            previous = elapsed
            yield sample


def replay_media_names(sample):
    home = sample.get("home") or {}
    return [value for value in [(sample.get("camera") or {}).get("path"),
        (sample.get("spatial") or {}).get("rgb"), (sample.get("spatial") or {}).get("depth"),
        (home.get("map") or {}).get("path"), home.get("telemetry_path")] if isinstance(value, str)]


def recorded_replay(batch_id, trial_index, root=None):
    root = root or RESULTS_ROOT
    directory, path, trial = replay_directory(batch_id, trial_index, root)
    frames, events, last_key, last_time, count = [], [], None, -1., 0
    previous_sample = None
    segment = 0
    prefix = f"/api/test-results/{batch_id}/replay/{trial_index}/media/"
    for sample in replay_samples(path):
        count += 1
        home = sample.get("home") or {}
        camera = sample.get("camera") or {}
        spatial = sample.get("spatial") or {}
        previous_home = (previous_sample or {}).get("home") or {}
        discontinuity = bool(previous_sample and (sample.get("manual_placements") != previous_sample.get("manual_placements")
            or (home.get("map") or {}).get("map_id") != (previous_home.get("map") or {}).get("map_id")
            or (home.get("localization") or {}).get("status") != (previous_home.get("localization") or {}).get("status")
            or home.get("map_from_odometry_m_rad") != previous_home.get("map_from_odometry_m_rad")))
        if discontinuity:
            segment += 1
        gap = sample["wall_s"] - previous_sample["wall_s"] if previous_sample else 0.
        contact_started = bool(sample.get("collisions")) and not bool((previous_sample or {}).get("collisions"))
        names = replay_media_names(sample)
        def media_url(name):
            return prefix + name.removeprefix("media/") if name in names and re.fullmatch(r"media/[a-z]+-[0-9]+(?:-depth)?\.(?:png|json)", name) else None
        key = (camera.get("path"), home.get("telemetry_path"))
        frame = {"wall_s": sample["wall_s"], "simulated_s": number(sample.get("simulated_s")),
            "index": count, "segment": segment, "activity": text(sample.get("activity")), "status": text(sample.get("physics_status")),
            "odometry_m_rad": scene_vector(sample["odometry_m_rad"], 3) if sample.get("odometry_m_rad") else None,
            "camera_url": media_url(camera.get("path")), "camera_simulated_s": number(camera.get("simulated_s")),
            "depth_url": media_url(spatial.get("depth")), "depth_simulated_s": number(spatial.get("simulated_s")),
            "map_url": media_url((home.get("map") or {}).get("path")),
            "telemetry_url": media_url(home.get("telemetry_path")), "home": home or None,
            "assisted": bool(sample.get("operator_assisted") or sample.get("manual_placements")),
            "contact": bool(sample.get("collisions"))}
        if key != last_key or sample["wall_s"] - last_time >= .5 or sample.get("home_events") or contact_started or discontinuity:
            if len(frames) >= 6000:
                raise ValueError("Replay exceeds 6000 keyframes")
            frames.append(frame)
            last_time, last_key = sample["wall_s"], key
        for event in sample.get("home_events", []):
            if len(events) < 2000:
                events.append({**{key: event.get(key) for key in ("stage", "localization", "task_id", "status", "segments", "retries", "stop_revision")},
                    "id": len(events) + 1,
                    "reason": text(event.get("reason")), "wall_s": sample["wall_s"]})
        for reason in (["Contact detected"] if contact_started else []) + (["Pose continuity changed"] if discontinuity else []) + ([f"Recording gap: {gap:.2f} s"] if gap > 1. else []):
            if len(events) < 2000:
                events.append({"id": len(events) + 1, "wall_s": sample["wall_s"], "stage": "recording", "status": "diagnostic", "reason": reason})
        previous_sample = sample
    if not count:
        raise ValueError("Empty replay")
    if frames[-1]["wall_s"] != frame["wall_s"]:
        frames.append(frame)
    return {"schema_version": 1, "frames": frames, "events": events, "sample_count": count,
        "evidence": trial["evidence"], "recording_complete": trial["recording_complete"],
        "spatial_progress": trial["spatial_progress"], "evaluation_only": True,
        "clock": "recording_monotonic_wall_seconds", "image_policy": "latest recorded past image; no future interpolation"}


def recorded_replay_media(batch_id, trial_index, name, root=None):
    import base64
    import zlib
    root = root or RESULTS_ROOT
    if not re.fullmatch(r"[a-z]+-[0-9]+(?:-depth)?\.(?:png|json)", name):
        raise ValueError("Invalid replay media name")
    directory, trajectory, _ = replay_directory(batch_id, trial_index, root)
    relative = "media/" + name
    if not any(relative in replay_media_names(sample) for sample in replay_samples(trajectory)):
        raise ValueError("Media is not referenced by the recording")
    path = directory / relative
    if not local_file(path, root) or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Recorded media unavailable")
    if name.endswith(".png"):
        image = path.read_bytes()
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Invalid camera PNG")
        return image, "image/png"
    data = read_json(path, root)
    if name.startswith("home-"):
        if data.get("encoding") != "int8_cells_then_packbits_visited_zlib_base64" or data.get("width") != 400 or data.get("height") != 400:
            raise ValueError("Unsupported recorded map")
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(base64.b64decode(data["data"], validate=True), 180001)
        except zlib.error as error:
            raise ValueError("Corrupt recorded map compression") from error
        if len(raw) != 180000 or not decoder.eof or decoder.unused_data or hashlib.sha256(raw).hexdigest() != data["sha256"]:
            raise ValueError("Corrupt recorded map")
        if not set(raw[:160000]).issubset({0, 100, 255}):
            raise ValueError("Invalid occupancy cells")
        data = {**{key: data[key] for key in ("map_id", "revision", "name", "frame", "width", "height", "resolution_m", "origin_m", "places", "edges", "sha256")},
            "cells": [value if value < 128 else value - 256 for value in raw[:160000]],
            "visited_indices": [index for index in range(160000) if raw[160000 + index // 8] & (1 << (7 - index % 8))]}
    elif not name.startswith("telemetry-"):
        raise ValueError("Unsupported replay data")
    return data, "application/json"