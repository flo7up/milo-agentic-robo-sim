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
def test_maze_requires_whole_robot_outside_and_uninterrupted_stationary_dwell(identifier, start, bay):
    progress = ChallengeProgress(get_challenge(identifier))
    inside, threshold, partial = ([bay[0] - offset, bay[1]] for offset in (1.8, .9, .5))
    elapsed = 0.
    for center, invalid in ((start, {}), (inside, {}), (threshold, {}),
                            (partial, {}), (bay, {"speed": .1}),
                            (bay, {"angular_speed": .2}), (bay, {"grounded": False})):
        elapsed += 2
        assert progress.update({"robot": {**robot_measurement(center), **invalid}}, set(), elapsed)["status"] == "in_progress"
    parked = {"robot": robot_measurement(bay)}
    assert progress.update(parked, set(), elapsed + .5)["status"] == "in_progress"
    assert progress.update(parked, set(), elapsed + .5)["status"] == "in_progress"
    assert progress.update({"robot": robot_measurement(threshold)}, set(), elapsed + .6)["status"] == "in_progress"
    assert progress.update(parked, set(), elapsed + 1.1)["status"] == "in_progress"
    assert progress.update(parked, set(), elapsed + 1.7)["status"] == "completed"
    assert progress.update({"robot": robot_measurement(threshold)}, set(), elapsed + 1.8)["status"] == "in_progress"


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
