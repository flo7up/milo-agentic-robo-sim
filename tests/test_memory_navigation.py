import asyncio
import base64
from io import BytesIO
import json
import math

import numpy as np
import pytest
from PIL import Image

from backend.agent import AgentController, AgentStart, FoundryConfig
from backend.challenges import get_challenge
from backend.continuous_navigation import ContinuousScan
from backend.home_mapping import MapStore
from backend.home_mission import HomeMission, HomeRequest
from backend.memory_session import configure_memory, lookup_memory, persist_memory, record_observation
from backend.worker import SimulationWorker
from tests.test_agent import model_response


def tv_bounds(image, details=None):
    if isinstance(image, str):
        image = base64.b64decode(image.split(",", 1)[1])
    pixels = np.asarray(Image.open(BytesIO(image))).astype(float)
    rows, columns = np.where((pixels[:, :, 0] > 100)
        & (pixels[:, :, 0] > pixels[:, :, 1] * 1.8)
        & (pixels[:, :, 0] > pixels[:, :, 2] * 1.8))
    if details is not None:
        details["red_pixels"] = len(rows)
    assert len(rows) > 8
    height, width = pixels.shape[:2]
    return [float(columns.min() / width), float(rows.min() / height),
        float((columns.max() + 1) / width), float((rows.max() + 1) / height)]


async def test_repeated_tv_observations_update_one_memory_entity(tmp_path):
    challenge = get_challenge("park").model_copy(update={"objectives": [], "objects": [
        {"name": "tv_screen", "size": [.12, .8, .6], "position": [1.5, 0., .65], "color": [.9, .03, .03, 1.]},
    ]})
    worker = SimulationWorker(pace=False, challenge=challenge, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "repeated-tv.sqlite3"))
        await configure_memory(worker, instance_id="repeated-tv-fixture")
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": worker.epoch}
        await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
        await worker.home_command(HomeRequest(**identity, action="start_mapping"))
        paired = await worker.call(lambda sim: next(reversed(worker.spatial_frames.values()))[:2])
        bounds = tv_bounds(paired[1])
        first = await record_observation(worker, context_id=worker.memory.scope.context_id, kind="object", label="TV",
            description="First current-camera TV evidence", bounds=bounds, selected_evidence=paired, source="fixture")
        def capture_newer(sim):
            worker._sample_spatial(force=True)
            worker._sample_spatial(force=True)
            return next(reversed(worker.spatial_frames.values()))[:2]
        newer = await worker.call(capture_newer)
        assert newer[0].sequence > paired[0].sequence
        shifted = [bounds[0] + .005, bounds[1], bounds[2], bounds[3]]
        second = await record_observation(worker, context_id=worker.memory.scope.context_id, kind="object", label="tv",
            description="Second current-camera TV evidence", bounds=shifted, selected_evidence=newer, source="fixture")
        assert first["observation_id"] != second["observation_id"]
        assert first["entity_id"] == second["entity_id"] and first["place_id"] == second["place_id"]
        assert worker.memory.summary["counts"]["objects"] == 1
        assert worker.memory.summary["objects"][0]["observation_count"] == 2
        assert len(worker.home_mission.home.places) == 1
    finally:
        await worker.close()


@pytest.mark.parametrize(("kind", "label", "lookup_action", "place_source"), [
    pytest.param("room", "Kitchen", "lookup_room", "camera_room_hypothesis", id="remembered-kitchen"),
    pytest.param("object", "TV", "find_object_sightings", "object_observation_viewpoint", id="remembered-tv"),
])
async def test_reloaded_memory_navigates_to_remembered_viewpoint(
        tmp_path, record_property, kind, label, lookup_action, place_source):
    challenge = get_challenge("park").model_copy(update={"objectives": [], "objects": [
        {"name": "back_wall", "size": [.1, 6., 1.], "position": [2.9, 0., .5], "color": [.7, .7, .7, 1.]},
        {"name": "front_wall", "size": [.1, 6., 1.], "position": [-2.9, 0., .5], "color": [.7, .7, .7, 1.]},
        {"name": "left_wall", "size": [5.8, .1, 1.], "position": [0., 2.9, .5], "color": [.7, .7, .7, 1.]},
        {"name": "right_wall", "size": [5.8, .1, 1.], "position": [0., -2.9, .5], "color": [.7, .7, .7, 1.]},
        {"name": "localization_landmark_tall", "size": [.35, .45, .9], "position": [-1.2, .9, .45], "color": [.3, .5, .7, 1.]},
        {"name": "localization_landmark_low", "size": [.55, .3, .4], "position": [1.4, -1.1, .2], "color": [.7, .5, .3, 1.]},
    ]})
    worker = SimulationWorker(pace=False, challenge=challenge, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "remembered-destinations.sqlite3"))
        await configure_memory(worker, instance_id="remembered-destinations-fixture")
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": worker.epoch}
        await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
        await worker.home_command(HomeRequest(**identity, action="start_mapping"))
        await worker.home_command(HomeRequest(**identity, action="add_place", name="Home"))
        home = next(place for place in worker.home_mission.home.places if place["name"] == "Home")

        await worker.home_command(HomeRequest(**identity, action="guided_to", pose_m_rad=[.65, 0., 0.], time_budget=45.))
        async with asyncio.timeout(50.):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert worker.home_mission.task["status"] == "completed", worker.home_mission.task
        remembered = await record_observation(worker, context_id=worker.memory.scope.context_id, kind=kind,
            label=label, description=f"Scripted current-camera evidence for {label}", source="fixture")
        assert remembered["place_id"] and not remembered["requires_revalidation"]
        remembered_pose = remembered["pose_m_rad"]

        await worker.home_command(HomeRequest(**identity, action="save_map", name="Remembered destinations"))
        await persist_memory(worker)
        profile_id = worker.memory.scope.profile_id
        await worker.home_command(HomeRequest(**identity, action="navigate_to", place_id=home["place_id"], time_budget=45.))
        async with asyncio.timeout(50.):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert worker.home_mission.task["status"] == "completed", worker.home_mission.task
        await persist_memory(worker)

        await configure_memory(worker, fresh=True, name="Temporary empty profile")
        await configure_memory(worker, profile_id=profile_id)
        assert worker.home_mission.localization["status"] == "unlocalized"
        await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
        await worker.home_command(HomeRequest(**identity, action="localize", place_id=home["place_id"]))
        lookup = await lookup_memory(worker, lookup_action, label)
        selected = next(item for item in lookup["records"] if item["place_id"] == remembered["place_id"])
        assert not selected["requires_revalidation"]
        place = next(item for item in worker.home_mission.home.places if item["place_id"] == selected["place_id"])
        assert place["source"] == place_source

        departure = list(worker.home_mission.pose)
        await worker.home_command(HomeRequest(**identity, action="navigate_to", place_id=selected["place_id"], time_budget=45.))
        async with asyncio.timeout(50.):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert worker.home_mission.task["status"] == "completed", worker.home_mission.task
        arrival = list(worker.home_mission.pose)
        position_error_m = math.dist(arrival[:2], remembered_pose[:2])
        travel_m = math.dist(departure[:2], arrival[:2])
        assert position_error_m <= .15 and travel_m >= .45
        assert not worker.sim.proximity_sensors().collisions
        record_property("memory_navigation", json.dumps({"evidence": "scripted_test", "kind": kind,
            "label": label, "profile_reloaded": True, "lookup_action": lookup_action,
            "travel_m": travel_m, "position_error_m": position_error_m, "sampled_contacts": 0}))
    finally:
        await worker.close()


async def test_reloaded_tv_memory_stages_fresh_object_verification(tmp_path, record_property):
    challenge = get_challenge("park").model_copy(update={"objectives": [], "objects": [
        {"name": "back_wall", "size": [.1, 6., 1.], "position": [2.9, 0., .5], "color": [.7, .7, .7, 1.]},
        {"name": "front_wall", "size": [.1, 6., 1.], "position": [-2.9, 0., .5], "color": [.7, .7, .7, 1.]},
        {"name": "left_wall", "size": [5.8, .1, 1.], "position": [0., 2.9, .5], "color": [.7, .7, .7, 1.]},
        {"name": "right_wall", "size": [5.8, .1, 1.], "position": [0., -2.9, .5], "color": [.7, .7, .7, 1.]},
        {"name": "localization_landmark_tall", "size": [.35, .45, .9], "position": [-1.2, .9, .45], "color": [.3, .5, .7, 1.]},
        {"name": "localization_landmark_low", "size": [.55, .3, .4], "position": [1.4, -1.1, .2], "color": [.7, .5, .3, 1.]},
        {"name": "tv_screen", "size": [.12, .8, .6], "position": [1.75, 0., .65], "color": [.9, .03, .03, 1.]},
    ]})
    worker = SimulationWorker(pace=True, challenge=challenge, rendering="enhanced")
    actions = []
    staging = {}

    class RecallModel:
        async def respond(self, profile, reasoning, goal, inputs):
            payload = json.loads(inputs[-1]["content"][0]["text"])
            current_image = next(part["image_url"] for part in inputs[-1]["content"] if part["type"] == "input_image")
            if not actions:
                decision = {"action": "plan", "plan": {"kind": "object", "target": "TV", "return_home": False}}
            elif actions[-1] == "plan":
                assert "find_object_sightings" in payload["available_actions"]
                decision = {"action": "find_object_sightings", "memory_query": "TV"}
            elif actions[-1] == "find_object_sightings":
                sighting = next(item for item in payload["last_execution"]["records"]
                    if item.get("place_id") and not item.get("requires_revalidation"))
                place = next(item for item in payload["observation"]["spatial"]["places"]
                    if item["place_id"] == sighting["place_id"] and item.get("reachable"))
                decision = {"action": "navigate_place", "place_id": place["place_id"]}
            elif actions[-1] == "navigate_place":
                staging.update(odometry_m_rad=payload["observation"]["odometry_m_rad"], verified=True,
                    receipts=payload["mission"]["receipts"])
                assert "target" not in payload["mission"]["receipts"]
                decision = {"action": "select_object", "object_label": "TV", "object_bounds": tv_bounds(current_image, staging)}
            elif actions[-1] == "select_object":
                selected = payload["object_goal"]
                assert selected and not selected["verified"]
                decision = {"action": "approach_object", "object_goal_id": selected["goal_id"]}
            else:
                selected = payload["object_goal"]
                assert actions[-1] == "approach_object" and selected["selected_approach"]
                decision = {"action": "verify_object", "object_goal_id": selected["goal_id"],
                    "object_bounds": tv_bounds(current_image)}
            actions.append(decision["action"])
            return model_response("guide_mission", json.dumps(decision), call_id=f"memory-tv-{len(actions)}")

        async def close(self):
            pass

    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: RecallModel())
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "tv-recall.sqlite3"))
        await configure_memory(worker, instance_id="tv-recall-fixture")
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": worker.epoch}
        await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
        await worker.home_command(HomeRequest(**identity, action="start_mapping"))
        await worker.home_command(HomeRequest(**identity, action="add_place", name="Home"))
        home = next(place for place in worker.home_mission.home.places if place["name"] == "Home")
        await worker.home_command(HomeRequest(**identity, action="guided_to", pose_m_rad=[.55, 0., 0.], time_budget=45.))
        async with asyncio.timeout(50.):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert worker.home_mission.task["status"] == "completed", worker.home_mission.task
        paired = await worker.call(lambda sim: next(reversed(worker.spatial_frames.values()))[:2])
        remembered = await record_observation(worker, context_id=worker.memory.scope.context_id, kind="object", label="TV",
            description="Scripted current-camera TV evidence", bounds=tv_bounds(paired[1]), selected_evidence=paired, source="fixture")
        staging["remembered_pose"] = remembered["pose_m_rad"]
        await worker.home_command(HomeRequest(**identity, action="save_map", name="Remembered TV"))
        await persist_memory(worker)
        profile_id = worker.memory.scope.profile_id
        await worker.home_command(HomeRequest(**identity, action="navigate_to", place_id=home["place_id"], time_budget=45.))
        async with asyncio.timeout(50.):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert worker.home_mission.task["status"] == "completed", worker.home_mission.task
        await persist_memory(worker)
        await worker.home_command(HomeRequest(**identity, action="add_place", name="Reload anchor",
            pose_m_rad=list(worker.home_mission.pose)))
        reload_anchor = next(place for place in worker.home_mission.home.places if place["name"] == "Reload anchor")
        await persist_memory(worker)
        await configure_memory(worker, fresh=True, name="Temporary TV profile")
        await configure_memory(worker, profile_id=profile_id)
        await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
        await worker.home_command(HomeRequest(**identity, action="localize", place_id=reload_anchor["place_id"]))
        assert worker.home_mission.localization["status"] == "localized"

        controller.start(worker, AgentStart(**identity, goal="Go to the remembered TV and verify it from a current image.",
            execution_mode="luna_continuous", unified_mission=True, images_per_request=2, map_context=True,
            mission_budget_s=180., max_turns=6, max_model_requests=6, max_model_tokens=100000))
        await asyncio.wait_for(controller.task, 195.)
        assert controller.state["error"] is None, {"actions": actions, "staging": staging}
        assert controller.mission.phase == "completed" and "target" in controller.mission.receipts
        assert actions == ["plan", "find_object_sightings", "navigate_place", "select_object", "approach_object", "verify_object"]
        assert staging["verified"]
        assert worker.object_goal and worker.object_goal.verified
        await worker.memory.flush()
        assert len(worker.memory.summary["objects"]) == 1
        assert worker.memory.summary["objects"][0]["entity_id"] == remembered["entity_id"]
        assert worker.memory.summary["objects"][0]["observation_count"] >= 2
        assert not worker.sim.proximity_sensors().collisions
        record_property("tv_memory_recall", json.dumps({"evidence": "scripted_test", "profile_reloaded": True,
            "actions": actions, "staging_verified": True, "fresh_verification": True,
            "sampled_contacts": 0}))
    finally:
        await controller.halt()
        await worker.close()