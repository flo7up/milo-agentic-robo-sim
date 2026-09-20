import copy
import asyncio
import json
import math
import time

import numpy as np
import pytest
from pydantic import ValidationError

from backend.home_mapping import HomeMap, MapStore
from backend.passage_memory import PassageMemory


def observed_map():
    home = HomeMap("observed-test")
    home.evidence[150:251, 150:251] = -2
    return home


def walk(home, points, start=0.):
    now = start
    for a, b in zip(points, points[1:]):
        for point in np.linspace(a, b, max(2, math.ceil(math.dist(a, b)/.15)+1)):
            now += .1
            home.passage_memory.observe([*point, 0.], now, home.observed_connection)
    return now


def loop(home):
    return walk(home, [[0., 0.], [2., 0.], [2., 2.], [0., 2.], [0., 0.]])


def test_measured_loop_retains_stable_places_and_does_not_count_dwell_as_visits():
    home = observed_map()
    now = loop(home)
    memory = home.passage_memory
    assert memory.data.loops >= 1 and memory.loop_active
    assert memory.current == "P1" and memory.places["P1"].visits == 2
    before = memory.document()
    for index in range(100):
        memory.observe([.01, 0., index/10.], now+index/10., home.observed_connection)
    assert memory.document() == before
    assert memory.summary(home.identity)["motion_authorized"] is False
    assert sum(edge.distance_m for edge in memory.edges.values()) > 7.


@pytest.mark.parametrize("fault", ["jump", "gap", "wall", "unknown", "nan"])
def test_history_never_invents_a_passage_across_discontinuous_or_unobserved_motion(fault):
    home = observed_map()
    walk(home, [[0., 0.], [.4, 0.]])
    before = len(home.passage_memory.edges)
    point, at = [.6, 0.], 1.
    if fault == "jump":
        point = [3., 0.]
    elif fault == "gap":
        at = 10.
    elif fault in {"wall", "unknown"}:
        home.evidence[:, 205] = 4 if fault == "wall" else 0
    else:
        point = [float("nan"), 0.]
    home.passage_memory.observe([*point, 0.], at, home.observed_connection)
    assert len(home.passage_memory.edges) == before
    assert not home.passage_memory.loop_active


def test_wall_separates_close_places_and_unknown_space_is_not_memory_evidence():
    home = observed_map()
    memory = home.passage_memory
    memory.observe([0., 0., 0.], 1., home.observed_connection)
    home.evidence[202, :] = 5
    memory.observe([0., .4, 0.], 1.1, home.observed_connection)
    assert len(memory.places) == 2 and not memory.edges
    assert memory.places["P1"].visits == 1
    blank = HomeMap("unknown")
    blank.passage_memory.observe([0., 0., 0.], 1., blank.observed_connection)
    assert not blank.passage_memory.places


def test_branch_attempt_survives_changing_frontier_id_and_keeps_unfinished_branches():
    home = observed_map()
    memory = home.passage_memory
    memory.observe([0., 0., 0.], 1., home.observed_connection)
    first = memory.annotate([{"frontier_id": "first", "position_m": [1.1, .1]}], [0., 0., 0.],
        home.observed_connection, home.known_near)[0]
    home.mark_frontier("first")
    second = memory.annotate([{"frontier_id": "shifted", "position_m": [1.3, .15]},
        {"frontier_id": "other", "position_m": [0., 1.3]}], [0., 0., 0.], home.observed_connection, home.known_near)
    assert second[0]["passage_id"] == first["passage_id"]
    assert second[0]["passage_attempts"] == second[0]["passage_no_progress"] == 1
    assert second[1]["passage_attempts"] == 0 and second[1]["passage_last_result"] == "untried"
    home.mark_frontier("shifted", "blocked")
    branch = memory.branches[first["passage_id"]]
    assert branch.attempts == 2 and branch.last_result == "blocked"
    assert "dead end" in memory.summary(home.identity)["caution"]


def test_persistence_retains_history_beyond_prompt_window_without_resuming_old_motion(tmp_path):
    home = HomeMap("long-history")
    home.evidence[:] = -2
    points = [[float(x), float(y)] for y in range(-15, 16, 3)
        for x in (range(-15, 16) if (y+15)//3 % 2 == 0 else range(15, -16, -1))]
    walk(home, points)
    assert len(home.passage_memory.places) > 128
    store = MapStore(tmp_path / "map.sqlite3")
    home.scan_count = 1
    before = copy.deepcopy(home.passage_memory.document())
    store.save(home, "Measured passages")
    restored = store.load(home.identity, home.environment_id)
    assert restored.passage_memory.document() == before
    assert restored.passage_memory.current is None and not restored.passage_memory.recent
    assert not restored.passage_memory.offers
    assert "passage_memory" not in HomeMap("fresh").document()


@pytest.mark.parametrize("fault", ["foreign_edge", "duplicate_id", "nonfinite", "oversized"])
def test_corrupt_saved_history_is_rejected(fault):
    home = observed_map()
    loop(home)
    data = home.passage_memory.document()
    if fault == "foreign_edge":
        data["passages"][0]["end"] = "P9999"
    elif fault == "duplicate_id":
        data["places"][1]["id"] = "P1"
    elif fault == "nonfinite":
        data["places"][0]["position_m"][0] = float("nan")
    else:
        data["places"] *= 100
    with pytest.raises(ValidationError):
        PassageMemory(data)


def test_changed_map_frame_cannot_reuse_old_places_or_branch_penalties():
    home = observed_map()
    loop(home)
    assert home.passage_memory.data.loops
    home.frame_revision = "new-alignment"
    candidates = home.exploration_frontiers([0., 0., 0.], .3)
    assert candidates and not home.passage_memory.places and not home.passage_memory.data.loops
    assert home.passage_memory.data.frame_revision == "new-alignment"


def test_loop_changes_frontier_choice_but_preserves_backtracking_and_clearance():
    home = observed_map()
    pose = [0., 0., 0.]
    baseline = home.frontiers(pose, .3)
    assert home.exploration_frontiers(pose, .3) == [dict(item, destination_visits=0) for item in baseline]
    loop(home)
    candidates = home.exploration_frontiers(pose, .3)
    assert candidates and candidates[0]["selection_basis"] == "new_observed_area_after_measured_loop"
    assert candidates[0]["destination_visits"] == 0
    for candidate in candidates:
        path = home.route(pose[:2], candidate["position_m"], .3)
        indices = home.indices(path)
        assert home.allowed(.3)[indices[:, 1], indices[:, 0]].all()
    excluded = candidates[0]["frontier_id"]
    assert all(item["frontier_id"] != excluded for item in home.exploration_frontiers(pose, .3, excluded=[excluded]))
    assert home.exploration_frontiers(pose, .3, obstacles=[[0., 0.]]) == []
    equal_options = [{"frontier_id": "repeat", "position_m": [1., 0.], "search_score": 5.},
        {"frontier_id": "new", "position_m": [-3., 0.], "search_score": 4.}]
    ordered = home.passage_memory.annotate(equal_options, pose, home.observed_connection, home.known_near, recovery=True)
    assert ordered[0]["frontier_id"] == "new" and ordered[1]["destination_visits"] > 0
    # Heavy revisit penalties never remove the only route out of a checked branch.
    source = [{"frontier_id": "return", "position_m": [1., 0.], "search_score": 0.}]
    ranked = home.passage_memory.annotate(source, [2., 0., math.pi], home.observed_connection, home.known_near, recovery=True)
    assert len(ranked) == 1 and ranked[0]["frontier_id"] == "return"
    assert ranked[0]["destination_visits"] > 0


def test_compact_model_payload_includes_measured_memory_without_geometry():
    from backend.mission import Mission
    from backend.mission_supervisor import semantic_payload
    home = observed_map()
    loop(home)
    candidates = home.exploration_frontiers([0., 0., 0.], .3)
    mission = Mission("run", 0, 0, 0, deadline=100.)
    observation = {"spatial": {"frontiers": candidates, "passage_memory": home.passage_memory.summary(home.identity)}}
    payload = semantic_payload(observation, mission, {}, None, ["explore"])
    memory = payload["observation"]["spatial"]["passage_memory"]
    assert memory["loops_detected"] and memory["places_reached"] > 4
    assert payload["observation"]["spatial"]["frontiers"][0]["selection_basis"]
    assert not memory["motion_authorized"]
    assert all(word not in json.dumps(payload) for word in ("maze_complex", "initial_xy", "exit_bay", "obstacles"))


async def test_real_worker_supplies_travelled_passages_to_model_and_stops(tmp_path, monkeypatch):
    from backend.challenges import get_challenge
    from backend.home_mission import HomeMission
    from backend.mission_supervisor import MappedReviewOperation
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    worker = SimulationWorker(challenge=get_challenge("kitchen_bathroom"), rendering="enhanced", pace=True)
    replies = [{"action": "plan", "plan": {"kind": "room", "target": "Kitchen", "completion": "identify"}},
        {"action": "explore"}, {"action": "wait", "duration_s": .5}, {"action": "wait", "duration_s": .5}]
    model = ScriptedModel([model_response("guide_mission", json.dumps(reply), call_id=f"memory-{index}")
        for index, reply in enumerate(replies)])
    controller = controller_for(model)
    original_review = MappedReviewOperation.review
    async def after_passage(mapped):
        async with asyncio.timeout(35.):
            while not worker.home_mission.home.passage_memory.edges:
                result = await mapped.poll()
                assert result["task"]["status"] == "running", result
                await asyncio.sleep(.05)
        mapped.reviewed_at = time.monotonic()-20.
        return await original_review(mapped)
    monkeypatch.setattr(MappedReviewOperation, "review", after_passage)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "physical-memory.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=True, images_per_request=2, mission_budget_s=90., max_turns=4, max_model_requests=4))
        await asyncio.wait_for(controller.task, 100.)
        assert controller.state["error"] is None, controller.state["error"]
        payloads = [json.loads(inputs[-1]["content"][0]["text"]) for inputs in model.inputs]
        memory = payloads[2]["observation"]["spatial"]["passage_memory"]
        assert memory["places_reached"] >= 2 and memory["passages_travelled"] >= 1
        assert memory["source"] == "measured_localized_movement" and not memory["motion_authorized"]
        assert memory["map_id"] == payloads[2]["observation"]["spatial"]["map_id"]
        assert payloads[3]["observation"]["spatial"]["passage_memory"]["passages_travelled"] >= memory["passages_travelled"]
        assert not worker.sim.proximity_sensors().collisions and worker.latest["stopped"] and not controller.active
    finally:
        await controller.halt()
        await worker.close()
