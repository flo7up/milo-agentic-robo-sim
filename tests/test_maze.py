"""Scripted physics feasibility and scoring checks, not real-model maze results."""

import numpy as np
import pybullet as bullet
import pytest

from backend.challenges import ChallengeLoad, ChallengeProgress, get_challenge
from backend.simulation import BulletSimulation
from tests.test_challenges import drive_to, robot_measurement
from tests.test_simulation import command


def observed_geometry_graph(sim, side, origin):
    """Evaluator-only rays through physical walls; never supplied to a controller."""
    def point(cell):
        return [origin + 1.8 * cell[0], origin + 1.8 * cell[1], 1.]

    graph = {(x, y): set() for x in range(side) for y in range(side)}
    exits = []
    for cell in graph:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (cell[0] + dx, cell[1] + dy)
            if bullet.rayTest(point(cell), point(neighbor), physicsClientId=sim.client)[0][0] == -1:
                if neighbor in graph:
                    graph[cell].add(neighbor)
                else:
                    exits.append((cell, neighbor))
    return graph, exits


def cell_route(graph, start, goal):
    parents, pending = {start: None}, [start]
    for cell in pending:
        for neighbor in sorted(graph[cell]):
            if neighbor not in parents:
                parents[neighbor] = cell
                pending.append(neighbor)
    route = [goal]
    while route[-1] != start:
        route.append(parents[route[-1]])
    return route[::-1]


@pytest.mark.parametrize("identifier,side,origin,edges,junctions,dead_ends", [
    ("maze", 4, -2.7, 15, 3, 5), ("maze_complex", 6, -4.5, 37, 10, 8),
])
def test_maze_is_connected_with_branches_dead_ends_and_one_hidden_exit(identifier, side, origin, edges, junctions, dead_ends):
    challenge = get_challenge(identifier)
    assert ChallengeLoad(challenge_id=identifier).challenge_id == identifier
    with pytest.raises(ValueError, match="supports"):
        ChallengeLoad(challenge_id=identifier, environment="shared_apartment_v1")
    sim = BulletSimulation(challenge=challenge, width=160, height=120)
    try:
        objects = {item["name"]: item for item in sim.objects}
        assert len(objects) == len(sim.objects)
        assert not sim.proximity_sensors().collisions
        assert sim.challenge_status()["status"] == "in_progress"
        assert sim.observe().odometry_m_rad == [0, 0, 0]
        assert not {"objects", "initial_xy", "floor_size_m", "passages"} & challenge.public().keys()
        for name in ("maze_exit_post_south", "maze_exit_post_north", "maze_exit_header"):
            target = objects[name]
            assert not sim._target_in_camera(target["id"], np.array(target["position"]))
        assert "maze_wall_" not in sim.observe().model_dump_json()

        graph, exits = observed_geometry_graph(sim, side, origin)
        visited, pending = set(), [(0, 0)]
        while pending:
            cell = pending.pop()
            if cell not in visited:
                visited.add(cell)
                pending.extend(graph[cell] - visited)
        assert len(visited) == side * side
        assert sum(len(neighbors) for neighbors in graph.values()) == edges * 2
        assert sum(len(neighbors) >= 3 for neighbors in graph.values()) >= junctions
        assert sum(len(neighbors) == 1 for neighbors in graph.values()) >= dead_ends
        assert exits == [((side - 1, side - 1), (side, side - 1))]
        if identifier == "maze_complex":
            assert edges - len(visited) + 1 == 2  # Two independent loops.
            assert len(cell_route(graph, (0, 0), (5, 5))) == 15
    finally:
        sim.close()


@pytest.mark.parametrize("identifier,start,bay", [("maze", [-2.7, -2.7], [4.5, 2.7]),
                                                 ("maze_complex", [-4.5, -4.5], [6.3, 4.5])])
def test_maze_completes_on_grounded_center_arrival_and_retains_success(identifier, start, bay):
    progress = ChallengeProgress(get_challenge(identifier))
    def measured(center, **overrides):
        return {"robot": {**robot_measurement(center), "position_xy": center, **overrides}}

    # Seeing/crossing the doorway and merely overlapping a bay edge do not count.
    for center in (start, [bay[0] - .9, bay[1]], [bay[0] - .601, bay[1]],
                   [bay[0] + .601, bay[1]], [bay[0], bay[1] - .651], [bay[0], bay[1] + .651]):
        assert progress.update(measured(center), set())["status"] == "in_progress"
    assert progress.update(measured(bay, grounded=False), set())["status"] == "in_progress"
    # Entry is enough while moving, without whole-body containment or a wait.
    arrival = [bay[0] - .599, bay[1]]
    status = progress.update(measured(arrival, speed=.3, angular_speed=.2), set())
    assert status["status"] == "completed" and status["completed_objectives"] == 1
    assert progress.update(measured(start), set(), 10.) == status
    reset = ChallengeProgress(get_challenge(identifier))
    assert reset.update(measured(start), set())["status"] == "in_progress"


@pytest.mark.parametrize("identifier", ["maze", "maze_complex"])
async def test_maze_worker_stops_a_drive_on_arrival_and_revokes_late_motion(identifier):
    import asyncio
    from backend.contracts import Command
    from backend.worker import SimulationWorker

    challenge = get_challenge(identifier)
    bay = challenge.objectives[0].center
    challenge.initial_xy = [bay[0] - .9, bay[1]]  # Scripted fixture at the doorway.
    worker = SimulationWorker(challenge=challenge, rendering="tiny", pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        before = worker.stop_revision
        def drive(identity):
            return Command(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
                observation_seq=worker.latest["observation"]["seq"], action_id=identity,
                tool="drive_base", arguments={"linear_mps": .3, "angular_radps": 0, "duration_s": 2})
        result = await worker.execute(drive("enter-bay"), assisted=False)
        assert result.status == "cancelled" and result.actual_duration_s < 2., result
        assert worker.arrival_completed and worker.latest["stopped"]
        assert worker.latest["challenge"]["status"] == "completed"
        assert worker.stop_revision > before
        position = await worker.call(lambda sim: bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0])
        assert bay[0] - .6 <= position[0] < bay[0] - .5
        ticks = worker.sim.ticks
        assert (await worker.execute(drive("late-drive"), assisted=False)).status == "cancelled"
        assert worker.sim.ticks == ticks
    finally:
        await worker.close()


@pytest.mark.parametrize("unified", [False, True])
async def test_maze_arrival_cancels_pending_inference_and_reports_physics_success(tmp_path, unified):
    import asyncio
    from backend.contracts import Command
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings

    class LateModel(ScriptedModel):
        async def respond(self, *args):
            try:
                return await super().respond(*args)
            except asyncio.CancelledError:
                return model_response()  # A late decision must never move the robot.

    challenge = get_challenge("maze_complex")
    bay = challenge.objectives[0].center
    challenge.initial_xy = [bay[0] - .9, bay[1]]
    worker = SimulationWorker(challenge=challenge, rendering="tiny", pace=False)
    model = LateModel(["wait"])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        if unified:
            worker.home_mission = HomeMission(worker, MapStore(tmp_path / "arrival.sqlite3"))
        controller.start(worker, start_settings(worker, unified_mission=unified,
            execution_mode="luna_continuous" if unified else "single_step", compact_arms=True,
            max_turns=5, mission_budget_s=30, images_per_request=2))
        await asyncio.wait_for(model.entered.wait(), 20)
        # Scripted physical crossing while inference is pending; no goal coordinates enter model inputs.
        await worker.execute(Command(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
            observation_seq=worker.latest["observation"]["seq"], action_id="inference-crossing", tool="drive_base",
            arguments={"linear_mps": .3, "angular_radps": 0, "duration_s": 2}), assisted=False)
        await asyncio.wait_for(controller.task, 5)
        assert worker.latest["challenge"]["status"] == "completed"
        assert controller.state["phase"] == "completed" and controller.state["error"] is None
        assert controller.state["outcome"]["kind"] == "completed"
        assert controller.state["outcome"]["source"] == "physics"
        assert model.closed and len(model.inputs) == 1 and not controller.active
        assert not controller.state["auto_wake"] and controller.idle_task is None
        if unified:
            assert controller.mission.phase == "completed"
        ticks = worker.sim.ticks
        await asyncio.sleep(.1)
        assert worker.sim.ticks == ticks
    finally:
        await controller.halt()
        await worker.close()


def test_complex_maze_dead_end_backtracking_and_long_exit_route_are_physically_feasible():
    sim = BulletSimulation(challenge=get_challenge("maze_complex"), width=160, height=120)
    try:
        graph, _ = observed_geometry_graph(sim, 6, -4.5)
        dead_end = (3, 4)
        assert len(graph[dead_end]) == 1
        route = cell_route(graph, (0, 0), dead_end) + cell_route(graph, dead_end, (5, 5))[1:]
        for x, y in route[1:]:
            drive_to(sim, [-4.5 + x * 1.8, -4.5 + y * 1.8])
            assert sim.challenge_status()["status"] == "in_progress"
            assert not sim.proximity_sensors().collisions
        drive_to(sim, [6.3, 4.5])
        assert command(sim, "wait", duration_s=1.1).status == "ok"
        assert sim.challenge_status()["status"] == "completed"
        assert sim.path_length > 29
        assert not sim.proximity_sensors().collisions
    finally:
        sim.close()


def test_maze_dead_end_backtracking_and_exit_are_physically_feasible():
    sim = BulletSimulation(challenge=get_challenge("maze"), width=160, height=120)
    try:
        # Privileged waypoints belong only to this scripted feasibility test.
        # Visit a dead end, backtrack, then traverse the exit with real wheel commands.
        for waypoint in ((-.9, -2.7), (.9, -2.7), (2.7, -2.7), (.9, -2.7), (.9, -.9),
                         (-.9, -.9), (.9, -.9), (.9, .9), (2.7, .9), (2.7, 2.7)):
            drive_to(sim, waypoint)
            assert sim.challenge_status()["status"] == "in_progress"
            assert not sim.proximity_sensors().collisions
        drive_to(sim, [4.5, 2.7])
        assert command(sim, "wait", duration_s=1.1).status == "ok"
        assert sim.challenge_status()["status"] == "completed"
        assert sim.path_length > 19
        assert not sim.proximity_sensors().collisions
    finally:
        sim.close()
    reset = BulletSimulation(challenge=get_challenge("maze"), width=160, height=120)
    try:
        assert reset.challenge_status()["completed_objectives"] == 0
        assert reset.challenge_status()["status"] == "in_progress"
    finally:
        reset.close()
