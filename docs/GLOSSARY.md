# Milo Robot Glossary

A plain-language reference for the simulator and the terms used in our discussions. Implementation notes describe the system as of 2026-09-12; a definition does not imply that the capability is already implemented.

## Recovery: What It Means Here

**Recovery is an attempt to get back to making useful progress after something goes wrong, while preserving the original task.** It can involve stopping, observing again, changing the approach, or requesting help. It does not mean pushing past a safety check or automatically restarting the simulation.

Example: the goal is to reach a doorway, but a chair blocks the approach.

1. **Detect:** notice the blocked path, insufficient clearance, or lack of progress.
2. **Make safe:** brake and discard movement that is no longer valid.
3. **Reassess:** obtain fresh observations and identify what changed.
4. **Adapt:** choose a different approach, or briefly retreat if that movement is safe.
5. **Verify:** check that the new action actually helps. Stop and request help if safe progress is not possible within the recovery budget.

The complete sequence is the intended recovery behavior, not a claim that Milo reliably performs all five steps today.

| Situation | Appropriate response | Current status |
| --- | --- | --- |
| A new obstacle blocks the continuous local path | Brake and replan within observed free space | The tracker can attempt two bounded same-goal replans without Luna. If no valid route exists, it remains stopped and requires fresh selection. |
| A proposed SmolVLA movement fails predictive clearance | Brake; Luna can inspect, choose another subgoal, or request a safe retreat | Available in Luna supervision, but successful recovery is not reliable. Actual contact and other hard safety faults end control. |
| The camera/map is too old | Stop and obtain fresh sensing before authorizing movement again | Freshness guards are implemented. A stopped continuous goal does not automatically resume when sensing returns. |
| The robot drifts slightly while following a clear path | Adjust steering from current measurements | Normal feedback correction, not usually called recovery. |
| The operator presses Stop or takes over | Cancel autonomous movement and leave control with the operator | Implemented. Recovery must never override an operator interruption or disconnected operator interface. |

**Retry** means attempting an action again. **Replanning** means calculating a different path. **Recovery** is the broader response that may use either. **Reset** starts a new episode and loses the current attempt; it is not autonomous recovery.

## Goals And Movement

| Term | Plain-language meaning |
| --- | --- |
| **Task / mission** | The overall objective, such as "find the bathroom" or "inspect the room and return to the charger." |
| **Goal / destination** | The outcome or location currently being pursued. A task can require several destinations. |
| **Subgoal / waypoint** | An intermediate target toward the overall task, such as a clear point before a doorway. Reaching it does not necessarily finish the task. |
| **Destination selection** | Manual mode uses a click on visible floor in paired RGB. Luna continuous mode chooses a numbered destination from depth-validated pixels or previously observed map floor. The local planner and tracker decide how to drive there. |
| **Path / route** | Where the robot should travel: a line or sequence of positions from its start to its destination. |
| **Trajectory** | A motion plan that also specifies how movement changes over time, such as positions or velocities with durations. A path alone does not specify speed. |
| **Planner** | Calculates a route around obstacles. The new local planner uses the observed map and the robot's size; it does not receive the simulator's hidden layout. |
| **Controller / path tracker** | Turns a path and current measurements into steering and speed commands. The planner chooses the route; the controller follows it. |
| **Closed-loop control / feedback** | Observe the result of movement and adjust the next command. This contrasts with executing an entire sequence without checking what happened. |
| **Continuous movement** | Updating motion before the previous short buffer ends, preserving velocity between updates. It still allows intentional stops at destinations and hazards. |
| **Differential drive** | Steering with two driven wheels: similar speeds move forward; different speeds turn the robot. Milo cannot move directly sideways. |
| **Curved motion / arc** | Driving forward while turning, using different nonzero wheel speeds. The current executor supports this; proving a scripted arc does not mean SmolVLA has learned it. |
| **Linear / angular velocity** | Forward or reverse speed in meters per second, and turning speed in radians per second. Positive turning commands rotate Milo left. |
| **Docking / precision parking** | The final approach and alignment needed to place the whole required footprint inside a destination, then stop. Seeing a floor marker in the image does not prove the wheels are on it. |
| **Autonomy** | How much the robot decides and executes without intervention. Luna can now choose continuous local destinations. One real kitchen run passed, but repeated autonomous room navigation remains unreliable. |

## Seeing And Mapping

| Term | Plain-language meaning |
| --- | --- |
| **RGB / head-camera view** | The normal color image from the robot's head. It is the robot's visual viewpoint, not the orbiting spectator view. |
| **Depth image / RGB-D** | An image of measured surface distances, or color and depth together. Milo's depth values are optical-axis distances in meters. Missing or invalid depth does not mean open space. |
| **Paired / synchronized frames** | RGB and depth captured together, with matching camera pose, timestamp and sequence number. This prevents measuring distance from a different view than the one displayed. |
| **Calibration / projection** | Calibration describes camera geometry and mounting. Projection uses that information to convert an image pixel plus depth into a spatial point. |
| **Point cloud** | A collection of measured 3D surface points reconstructed from depth. It represents visible surfaces, not a complete solid model of the room. |
| **Observed map / occupancy grid** | A grid recording floor and obstacles detected by the robot, with unobserved cells marked unknown. It is built from sensing, not copied from the spectator scene. |
| **Unknown space** | Space not sufficiently observed. It might be clear or occupied; the planner must not assume it is safe. The local controller has a documented, limited start-area allowance for near-body self-occlusion. |
| **Occlusion / self-occlusion** | Something blocks the camera's view. Self-occlusion means the robot's own body or arms are the obstruction. Filtering those pixels does not reveal what is behind them. |
| **Traversability** | Whether the robot can safely pass through a region, considering observations, its footprint and clearance. Visible floor alone is not enough. |
| **Footprint / clearance / inflation** | Footprint is the space the robot occupies, including projecting arms. Clearance is the gap around it. Inflation reserves room around obstacles and unknown boundaries so planning does not treat the robot as a dimensionless point. |
| **Pose / coordinate frame** | Pose is position plus orientation. A coordinate frame specifies the reference: "one meter forward from the robot" differs from "one meter along the map's x-axis." |
| **Odometry / drift** | Odometry estimates movement from wheel rotation. Its errors can accumulate, producing drift. The map's robot marker is based on that estimate, not perfect simulator position. |
| **Proprioception** | Measurements of the robot's own state, such as joint angles, wheel speeds and gripper opening. |
| **Localization / SLAM** | Localization estimates where the robot is. SLAM jointly builds a map and estimates position, often correcting accumulated drift. Milo currently uses wheel odometry; drift-correcting SLAM is not implemented. |
| **Frontier exploration / semantic search** | Frontier exploration investigates the boundary between known and unknown space. Semantic search looks for meaningful targets such as a bathroom or red chair. These are later autonomy milestones, not features of the current click-to-navigate controller. |

## Timing And Safety

| Term | Plain-language meaning |
| --- | --- |
| **Motion buffer / horizon** | A short amount of authorized future movement. The continuous controller refreshes a one-second buffer. It is not permission to keep moving indefinitely. |
| **Lookahead** | Checking where proposed motion would take the robot over a short future interval, before executing it. This is separate from finding the whole route. |
| **Buffer starvation** | The controller fails to supply the next valid movement in time. Milo stops rather than continuing an outdated command. This is different from an intentional arrival stop. |
| **Freshness / stale data** | Whether an observation is recent enough to trust for the current movement. Stale data can describe a world or robot pose that has changed. |
| **Lease / watchdog** | A lease is an expiry time on motion authorization. A watchdog checks conditions such as deadlines and sensing freshness, stopping the robot when they fail. |
| **Safety stop / clearance stop** | Braking because a required safety condition is missing. A predictive clearance stop can happen before contact; it is not evidence that a collision occurred. |
| **Latency / update rate / Hz** | Latency is how long a response takes; update rate is how often updates occur. Hz means updates per second. Physics integration, sensing, local control and model inference have different rates. |
| **Wall time / simulated time** | Wall time is real elapsed time. Simulated time advances when physics advances. Time spent waiting for a model does not necessarily move the simulation forward. |
| **Local-route limit / test budget** | The current continuous route is limited to 1.8 m and a 60-second skill deadline. The isolated model evaluator's default 30-minute test budget is a separate limit; it does not override local safety deadlines. |

## Models, Training And Evidence

| Term | Plain-language meaning |
| --- | --- |
| **Luna / supervisor** | The configured cloud model used for visual reasoning and goal decisions. It can select observed destinations for conventional continuous control or supervise the legacy SmolVLA primitive flow. |
| **SmolVLA / learned policy** | A local model that predicts actions from images, measured state and instructions. It is one possible controller component, not the whole robot system. |
| **Deterministic / conventional controller** | Explicit algorithms compute actions from inputs instead of generating them with a learned model. The new path tracker uses this approach; it still needs testing and safety checks. |
| **Inference / fine-tuning / checkpoint** | Inference uses a model to produce an answer or action. Fine-tuning changes its learned weights using examples. A checkpoint is a saved set of weights. Running more tests does not itself train the model. |
| **Demonstration / replay** | A demonstration records an example of performing a movement or task. Replay executes saved actions to check reproducibility. A successful scripted replay is not proof that a model can perform the task. |
| **Held-out test / generalization / regression** | Held-out cases are excluded from training. Generalization means working beyond the training examples. Regression tests check that a change has not broken existing behavior. Nearby variations alone do not prove broad generalization. |
| **Episode / run / reset** | An episode is one instance of the simulated scene. A run is a controller attempt within it. Reset creates a fresh episode; several attempts can otherwise occur in the same scene. |
| **Run chat / redirect** | A new operator instruction during navigation. It replaces the active goal, stops obsolete motion and replans in the same scene with the remaining budget. It does not run a second controller or reload resident model weights. |
| **Movement memory** | Bounded history of measured visits, inspected headings, executed movements and nearby failed actions. Both Luna modes receive it, including after a redirect. Reset, challenge change or manual placement clears location-based memory. |
| **Landmark assessment** | A model's interpretation of observed fixtures or rooms. Stored separately from measured movement because it can be wrong; it is not a verified room label. |
| **Physics-verified success / agent claim** | The independent physics scorer checks the challenge's actual conditions. An agent claim is the model saying it succeeded; that statement can be wrong. Arrival at a clicked waypoint is also separate from completing the loaded challenge. |
| **Ground truth / spectator state** | The simulator knows exact scene geometry and object positions for rendering and scoring. Those privileged values are not navigation/model observations. The internal physics safety checker can use scene geometry without giving it to the route planner. |

## Current Navigation Flows

- **Start LLM control / Continuous local control:** the default selector. Luna chooses observed destinations; a conventional tracker supplies continuous wheel commands within each route. Scanning and semantic decisions still pause motion. SmolVLA is not used.
- **Plan while moving:** optional experimental overlap. Luna proposes a compatible continuation while the current route runs. The worker revalidates it without resetting the original distance/time limits. Late replies, arrivals and Stop cannot restart motion.
- **Start LLM control / SmolVLA primitives:** Luna chooses bounded intentions; SmolVLA predicts wheel velocities. This remains experimental and pauses between movements.
- **Scan floor -> Select destination:** the operator chooses an observed floor point; a conventional local planner and tracker drive continuously to it. There are no Luna or SmolVLA calls in this mode, and blocked routes require another selection.

Luna-to-continuous integration and bounded opt-in planning overlap are implemented. Reliable room recognition and general recovery remain unfinished; one successful development run is not a reliability guarantee. The recorder and offline replay distinguish measured travel, inference waits and physical completion from model claims.

For implementation details, see [MECHANICS.md](MECHANICS.md). For training history, see [SMOLVLA.md](SMOLVLA.md). For measured outcomes and caveats, see [VALIDATION.md](VALIDATION.md).