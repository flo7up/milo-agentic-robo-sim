import asyncio
import json
import math

import pytest

from backend.challenges import get_challenge
from backend.continuous_navigation import ContinuousScan
from backend.home_mapping import MapStore
from backend.home_mission import HomeMission, HomeRequest
from backend.memory_session import configure_memory, lookup_memory, persist_memory, record_observation
from backend.worker import SimulationWorker


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