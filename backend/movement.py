import asyncio
import math
import time

import numpy as np
from pydantic import Field, model_validator
from typing import Literal

from backend.contracts import StrictModel


class MovementStep(StrictModel):
    kind: Literal["drive", "turn"]
    distance_m: float = Field(default=0., ge=-1.5, le=1.5, allow_inf_nan=False)
    angle_rad: float = Field(default=0., ge=-6*math.pi, le=6*math.pi, allow_inf_nan=False)

    @model_validator(mode="after")
    def one_axis(self):
        if self.kind == "drive" and (abs(self.distance_m) < .05 or self.angle_rad != 0.):
            raise ValueError("Drive steps require a signed distance of 0.05-1.5 m and no turn")
        if self.kind == "turn" and (abs(self.angle_rad) < .05 or self.distance_m != 0.):
            raise ValueError("Turn steps require a signed angle of 0.05-6*pi radians and no translation")
        return self


async def execute_movements(controller, worker, settings, mission):
    from backend.navigation import NAVIGATION_TOOLS
    from backend.continuous_supervisor import wait_stationary
    results = []
    origin = await worker.call(lambda sim: sim.odometry[:2].copy())
    for step in mission.plan.movements:
        controller._check_live(worker, settings)
        mission.check((worker.sim.run_id, worker.epoch, worker.stop_revision, worker.task_revision))
        await worker.begin_navigation(mission.authority[2])
        start = None
        def guard(sim):
            mission.check((sim.run_id, sim.epoch, worker.stop_revision, worker.task_revision))
            if sim.cancel.is_set():
                raise ValueError("Movement cancelled")
            worker._sample_spatial()
            home = worker.home_mission
            home.sample()
            home.require_localized()
            sensor = worker.mapped_depth_sensor
            observed = worker.spatial_map
            if (worker.spatial_error or sensor is None or not worker.spatial_enabled
                    or not 0 <= time.monotonic()-sensor.captured_at <= 1.
                    or observed is None or observed.captured_at is None or not 0 <= time.monotonic()-observed.captured_at <= observed.max_frame_age_s):
                raise ValueError("SPATIAL_STALE: movement requires fresh depth")
            pose = sim.odometry
            endpoint = start[:2] + step.distance_m*np.array([math.cos(start[2]), math.sin(start[2])])
            allowed = observed.traversable(origin, sim.robot_footprint()["radius_m"])
            points = np.linspace(pose[:2], endpoint, max(2, math.ceil(math.dist(pose[:2], endpoint)/.025)+1))
            if not observed.contains_path(allowed, points):
                raise ValueError("OBSERVED_PATH_BLOCKED: requested movement leaves observed footprint-clear space")
            delta = pose[:2]-start[:2]
            lateral = -math.sin(start[2])*delta[0]+math.cos(start[2])*delta[1]
            if (step.kind == "turn" and np.linalg.norm(delta) > .03) or (step.kind == "drive" and abs(lateral) > .03):
                raise ValueError("MOVEMENT_DRIFT: requested movement left its line or pivot")
        def apply(sim, name, arguments):
            observation = sim.observe(render=False)
            worker.navigation.observe(sim, observation)
            parsed = NAVIGATION_TOOLS[name].model_validate({"expected_revision": worker.navigation.revision, **arguments})
            worker.navigation.apply(sim, name, parsed, observation.seq)
        def begin(sim):
            nonlocal start
            start = sim.odometry.copy()
            worker._sample_spatial(force=True)
            worker.home_mission.sample(force=True)
            guard(sim)
            apply(sim, "begin_local_subgoal", {"goal": "Execute requested measured " + step.kind})
            worker.navigation.skill_deadline = min(worker.navigation.skill_deadline, mission.deadline)
            worker.movement_guard = guard
        try:
            await worker.call(begin)
            while True:
                controller._check_live(worker, settings)
                def update(sim):
                    guard(sim)
                    delta = sim.odometry-start
                    progress = (delta[0]*math.cos(start[2])+delta[1]*math.sin(start[2])) if step.kind == "drive" else delta[2]
                    remaining = (step.distance_m if step.kind == "drive" else step.angle_rad)-progress
                    if abs(remaining) <= (.02 if step.kind == "drive" else .035):
                        worker.navigation.cancel(sim, "Requested movement reached")
                        return True
                    linear = float(np.clip(.8*remaining, -.15, .15)) if step.kind == "drive" else 0.
                    angular = float(np.clip(-2*delta[2] if step.kind == "drive" else 2*remaining, -.5, .5))
                    apply(sim, "replace_motion_buffer", {"segments": [{"kind": "drive", "linear_mps": linear,
                        "angular_radps": angular, "duration_s": .5}]})
                    return False
                if await worker.call(update):
                    break
                await asyncio.sleep(.05)
            await wait_stationary(controller, worker, settings, .6)
            def verify(sim):
                guard(sim)
                delta = sim.odometry-start
                progress = float(delta[0]*math.cos(start[2])+delta[1]*math.sin(start[2])) if step.kind == "drive" else float(delta[2])
                error = abs((step.distance_m if step.kind == "drive" else step.angle_rad)-progress)
                if error > (.04 if step.kind == "drive" else .08):
                    raise ValueError("Movement did not settle at its requested distance or angle")
                return {"step": step.model_dump(), "measured": progress, "error": error, "source": "wheel_odometry"}
            result = await worker.call(verify)
            results.append(result)
            controller._trace("policy", "Requested movement step completed", result)
        finally:
            def release(sim):
                worker.movement_guard = None
                if worker.navigation:
                    worker.navigation.cancel(sim, "Movement step ended")
            await worker.call(release)
    return {"status": "movement_completed", "steps": results, "source": "wheel_odometry", "independent_evaluation": False}