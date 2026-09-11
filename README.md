# Milo: Embodied Robot Lab

An in-progress visual robotics research simulator. The mechanical backend and React operator interface are implemented and tested, including desktop and mobile browser checks. This is not a completed implementation of the five-milestone specification.

## Windows Setup

Prerequisites: PowerShell, Node.js 20.19+ (24.12.0 on the test machine), and `uv` on PATH. No cloud credentials are required for the mechanical bench.

```powershell
./scripts/setup-windows.ps1
./.runtime/env/python.exe -m pytest -q
./start.ps1 -BackendOnly
```

Backend API: http://127.0.0.1:8000/docs. The API and WebSocket server bind to loopback only. Do not expose this prototype to a network without authentication and additional hardening.

To launch the operator interface:

```powershell
npm --prefix frontend ci --registry=https://packagefeedproxy.microsoft.io/npm/
./start.ps1
```

On Microsoft-managed devices, use the CFS-protected feed above. Do not switch to the blocked public registries. The frontend lockfile uses the internal Microsoft tarball host returned by CFS.

The root launcher builds React and serves both the UI and API at http://127.0.0.1:8000. Use `-Port 8002` when 8000 is occupied; port 8001 is reserved for automated tests. `-SkipBuild` reuses a previously built frontend. No deployment or inference credentials are needed for manual operation.

To run the fine-tuned local navigation model in its separate test scene:

```powershell
./start.ps1 -NavigationTest -WhatIf
./start.ps1 -NavigationTest
```

This requires the separate Windows model environment and a completed navigation checkpoint. The default is `.runtime/navigation-stop-balanced-1500/checkpoint`, case 15, seed 716, with a 30-request limit. Override these with `-Checkpoint`, `-Case`, `-Seed`, and `-Requests`; use `-ModelPython` for another model interpreter. Results go to a fresh directory under `.runtime/navigation-tests`, or a new directory specified by `-Output`. The launcher prints the task outcome, report path, and camera timelapse path. `-WhatIf` checks prerequisites without starting inference or creating results.

`-NavigationTest` does not launch the browser, change an existing browser scene, or start the left-arm policy service. Use plain `./start.ps1` for the browser. Its Single step and Navigation plan modes use the selected LLM; Luna + SmolVLA still needs a separate compatible left-arm server. Navigation and browser-only flags cannot be combined.

For separate frontend development, start the backend and run `npm --prefix frontend run dev`; Vite proxies `/api` and WebSocket traffic to port 8000. After building, run `npm --prefix frontend test` with an installed Microsoft Edge browser. Playwright starts an isolated test server on port 8001 with real physics and scripted model replies; it does not use your live Foundry connection or reset the manual session on port 8000. Port 8001 must be free. Set `ROBOSIM_PYTHON` to an alternative simulator interpreter when needed.

## Watch the Robot Move

Run `./start.ps1` and open http://127.0.0.1:8000. Under **Manual control**, select **Base** and use the forward or turn buttons. **Resume manual control** releases a previous Stop. Movement is simulated at 240 Hz and played at wall-clock speed in the spectator view; a two-second motion takes approximately two seconds.

For model-directed movement, use the default **GPT-5.6 Luna** profile (`gpt-5.6-luna`), enter a goal such as "Move forward a short distance, turn left, then stop", and click **Start LLM control**. **GPT-5.4 Nano** (`gpt-5.4-nano`) is also available in the model selector for testing. Both require deployment access at the configured Foundry endpoint; the app does not create deployments. The **Exchange feed** shows inputs and actions. **Stop** interrupts motion; **Take manual control** also releases model ownership. Physics holds between commands and while the model thinks: this is not a continuously free-running simulation.

During manual, text-model, and voice-directed actions, the spectator receives poses every 12 physics ticks (20 Hz simulated) and the head-camera preview targets 10 frames per wall-clock second. Both update before an action finishes; actual throughput depends on the machine. The camera's **Live** counter and simulated timestamp identify the displayed frame. Slow downloads skip intermediate previews without accumulating requests. These operator previews do not advance the model's observation sequence or change its feedback interval; the exchange feed retains the exact images submitted to the model.

The status strip above the views stays visible while scrolling. **Robot running** is highlighted only while the worker executes a command, with the simulated-time counter alongside it. **Model thinking** and **Waiting for feedback** explicitly show that the robot is holding position. Idle, stopped, finished, voice, and disconnected states have distinct labels; after a disconnect the app reports the robot state as unavailable until fresh state arrives.

### Textures and Sensors

Scenes use local bitmap materials for concrete, plaster, wood flooring, tile, fabric, and stone. The same texture assets appear in the PyBullet head camera and Three.js spectator; the spectator also has soft shadows. These are original generated surface patterns, with no external texture downloads or changes to collision geometry.

The **Distance sensors** panel shows eight chassis-relative beams: front, rear, left, right, and four diagonals. Each measures up to 2 m from its mount, about 4 cm above the floor in the upright pose. **>2.00 m** means no hit within range; **Occluded** means the robot itself blocks that beam. Readings below 0.25 m are highlighted. Narrow beams can miss obstacles between them or above/below their height; they are not a full clearance map and do not automatically brake.

**Collision detected** shows the side and summed contact force in newtons. Normal floor/support contact and finger contacts with movable grasp objects are excluded; hitting a wall with an arm or finger still counts. Contact readings can clear after the robot brakes. Live operator readings update during actions, while Chat/Voice models receive the corresponding sensor snapshot only with their scheduled robot observation. Object names, IDs, and coordinates are never included in sensor readings.

### Task Outcomes

The sticky result band keeps the outcome visible above the simulation, including after the agent enters idle. **Task completed** is green only when the current scenario's physics checks pass, and lists the verified objective count. Episodes with manual placements are marked **Operator-assisted**. This verifies the scenario conditions, not an autonomous-model benchmark.

**Agent reports completion** and **Task cannot be completed** retain the agent's reason and identify it as an unverified report. **Task failed**, **Action blocked/cancelled**, **Run interrupted**, **Turn limit reached**, and **Agent error** have distinct labels and details. A text-only reply is **Response finished**, not proof of task success. Newer manual results replace older controller reports; reset clears them, and disconnect shows that the current outcome is unavailable. Scenario success reflects current conditions and clears if those conditions no longer hold.

### Position the Robot

In idle manual control, click or tap Milo in the spectator view to select it, then drag it to a new floor position and release. A selection outline and X/Y preview show the proposed placement. Dragging the background still orbits the spectator. The selection icon is also keyboard-accessible: arrow keys adjust the selected position by 0.1 m, Enter applies it, and Escape cancels.

A drop changes the real robot position only after server validation, then refreshes the head camera. It preserves heading, height, and joint pose and leaves scene objects in place. Obstacle overlaps, positions beyond the floor, stale drops, and placement while holding an assisted grasp are rejected. Take manual control and resume after Stop before repositioning; placement is unavailable during model control or motion.

Placement does not simulate driving, advance time, change encoder odometry, or consume travel energy. The spectator footer records **Manual placements** until reset. Challenge progress still reflects the scene, but a manually repositioned episode is operator-assisted, not an autonomous challenge result. Placement is not an LLM tool.

## Predefined Challenges

Choose a task from **Challenge** and click **Load challenge**. Ten scenarios are grouped into Navigation, Perception and Manipulation; the clinic and three new training grounds are marked Advanced. Each option creates a fresh physical scene, resets progress, and fills the robot goal automatically. Select your configured model and click **Start LLM control** to attempt it, or use the manual controls. Voice sessions receive the same public goal; you can ask Milo to solve the loaded challenge.

| Challenge | Goal | Completion |
| --- | --- | --- |
| Apartment Search | Explore a small flat and find the yellow cube on a pedestal. | Within 0.9 m, grounded and stationary, with the target visible in the head camera for one simulated second. |
| Kitchen to Bathroom | Identify the starting room from its fixtures, then navigate through the hall to the bathroom. | Entire base and wheels within the bathroom's clear arrival area, grounded and at rest; the room description is not automatically graded. |
| Park in the Bay | Navigate to the green bay beyond the yellow posts. | Entire base and wheels inside the bay, on the floor and at rest. |
| Tidy the Cube | Pick up the red cube and put it in the blue floor zone. | Cube lifted clear of the floor, then fully inside the zone, released and settled. |
| Color Sort | Put the red and blue cubes in their matching colored zones. | Both cubes individually lifted, released, fully contained, and settled in the correct zones. |
| Remember and Recharge | Remember the starting cyan charger, visit the orange survey zone behind a screen, then return when the battery is low. | Complete the survey, return to the original pad after the warning, and recharge to at least 90%. |
| Clinic Supply Delivery | Navigate reception, bypass the maintenance barrier, and enter treatment through diagnostics. | Grounded transit through two ordered checkpoints, then base/wheels fully parked in the green bay. No carried supplies are required or scored. |
| Warehouse Dispatch Circuit | Route around offset stock racks in a 10 x 8 m warehouse, following blue, orange, then green checkpoints. | Ordered grounded crossings, followed by full dispatch-bay parking for one simulated second. Scripted route exceeds 20 m. |
| Service Gallery Inspection | Search an 8 x 8 m equipment gallery behind occluding partitions; reject the red decoy. | Yellow target within 0.9 m, unobstructed head-camera visibility and a stationary one-second inspection. Scripted route exceeds 15 m. |
| Cluttered Assembly Workshop | Use both arms to sort small cubes among fixed divider blocks, parts racks and a workbench. | Both cubes lifted, released, fully inside matching squares and settled on the floor. |

The checklist uses actual physics measurements, not an LLM's claim. For pick-and-place tasks it records a lift while held by grasp assistance or real opposing finger contacts. Pushing a cube into a zone, holding it above the zone, placing it partly outside, or swapping colors does not count. Zones are visible floor markings with no collision surface. Cubes are used because they match the current validated gripper; spherical-object manipulation is not included in these presets.

**Reset episode** restarts the currently loaded task. Choose **Practice bench** to return to unscored practice. Loading or resetting interrupts text/voice control and invalidates previous-episode commands; connection configuration is preserved. Scene positions, grasp history, and scoring stay on the backend. Models receive only the public goal, their normal robot sensors, and head-camera images. The model may need to tilt or turn the head to inspect objects on the floor.

All ten tasks' physical completion checks have been satisfied by scripted commands through real physics, including both workshop pickups and the new long routes. This establishes feasibility, not autonomous room recognition or model success rates. Ordered-route intermediate checkpoints retain visit history; the final destination must still satisfy its current conditions, so leaving or moving clears completion. Transit does not count without floor contact. No run history or benchmark result is persisted.

### Testing Capabilities

Use **Single step** for general camera-guided tasks and manipulation. **Navigation plan** supports bounded navigation only: warehouse and inspection are useful tests of route selection, occlusion, progress and recovery. The workshop tests the canonical arm/gripper tools. The experimental SmolVLA checkpoint remains a left-arm-only policy trained on a different small fixture; it is not validated for these grounds or both-arm workshop operation. Loading a challenge does not train model weights or export demonstrations.

Start with Park, then Clinic, then Warehouse for navigation. Compare Apartment and Service Gallery for visual search. Compare Color Sort and Workshop for manipulation. Keep model, feedback interval, image count and retained context fixed for comparable attempts, reset before each run, and record actual completion, actions, contacts, elapsed time, tokens and any manual placement. The exchange feed can be copied before resetting. Report a manual-assisted result separately from an autonomous run.

The grounds are fixed and repeatable, not randomized or held-out benchmarks. Furniture is simplified rigid geometry; there are no pedestrians, moving doors or soft-body hazards. Workshop fixture avoidance and keeping the base parked are task instructions, not separate scored objectives; lift/release/settle is the measured success condition. Large spectator views fit the actual floor on desktop/mobile but remain privileged operator views, never model maps.

### Apartment Search

Select **Apartment Search** and click **Load challenge**. The roofless spectator view shows a modest three-room flat: an entry/living area with a sofa and two rooms reached through 1.3 m open doorways. A kitchen counter and colored room floors provide visual landmarks. The task has 100 model turns available by default. Doorways are permanently open; no door-opening tool is required.

Find the **yellow cube on a short pedestal**. A red cube in the other room is a distractor, and the yellow target is hidden from the initial head-camera view by a wall. The model receives the goal and its usual head-camera/sensor inputs, not a floor plan, target coordinates, or a target-bearing sensor.

Approach within 0.9 m and stop with the target clearly visible for one simulated second. Use **Wait** to advance the inspection while holding position. Completion requires a stationary base and head, floor contact, and unobstructed camera sight of the correct target. Looking away, remaining behind a wall, moving past it, or inspecting the red cube does not count. No pickup is needed. Manual drag placement remains available but marks the episode as operator-assisted.

### Kitchen to Bathroom

This separate apartment scenario starts Milo in the kitchen, facing a fridge/freezer, a sink with faucet, and a stove with burners and an oven. The bathroom has a toilet, basin with faucet, mirror-like panel, bathtub, towel, and tiled surfaces. These fixtures appear in both the robot's head camera and the roofless spectator view; they are physical scene geometry, not room-name overlays.

The goal asks the model to identify its starting room from the visible fixtures, navigate through the open doorways and hall, then describe the bathroom on arrival. The model receives no current-room label, floor plan, fixture identities, or destination coordinates. It may turn its head or base to inspect the rooms. The task defaults to 100 model turns, with permanently open 1.3 m doorways and no furniture manipulation required.

Stop fully inside the bathroom's clear floor area. Completion requires the entire base and wheels inside the private arrival region, floor contact, and rest; there is no colored destination marker. The check rejects stopping in the kitchen, hall, or partly in the doorway and clears after leaving. It verifies physical arrival only: room descriptions are visible model responses, not semantically graded answers. The fixtures are simplified, and the mirror-like panel is not reflective. Scripted wheel navigation and camera images are validated; autonomous model recognition and navigation remain unmeasured.

### Remember and Recharge

The robot starts on a cyan charging pad with a full battery and its camera angled toward the pad. A gray screen blocks the direct view between the charger and the orange survey area, with colored landmarks and room to drive around it. The agent must remember its starting location and route; no homing command, target coordinates, bearing, or distance-to-charger sensor is supplied.

Parking in the orange zone for one simulated second completes a scan that consumes 60 percentage points of charge. Normal activity also drains the battery. The robot then reports low battery at 35% or below and must navigate back under its own control. Charging occurs only when the entire base/wheels are stationary on the original pad with floor contact. Staying there from the start, returning before the survey, or parking on the wrong zone cannot complete the challenge.

Use the bounded `wait` tool to advance survey/charging time; `observe` alone does not advance physics. The manual **Wait** button uses the selected duration. Zero charge blocks further movement, so an empty robot stranded outside the charger requires an episode reset. Battery percentage, low-battery state, and charging state appear in the sensor panel and exchange feed. They are the only new model-visible battery data.

This task defaults to 100 model turns. With **Images per request** set to at least two, its text-model controller reserves a historical image slot for the first genuine camera/sensor observation when that image differs from the current view. The reference counts toward the selected image limit and is identified in the feed. At one image, only the current view is sent. Voice uses its existing conversation history. This fixed-layout exercise practices memory-guided navigation; it is not a held-out memory benchmark or a realistic hardware battery model.

## Chat and Voice Modes

Use the **Chat** and **Voice** tabs under **Agent interaction**. They are separate interaction modes with separate connection controls. Switching modes stops current robot control and cancels pending inference; leaving Voice releases recording and playback. Selecting Voice alone does not activate the microphone. The selected mode is shared by connected operator pages, and resetting the episode returns to Chat.

In **Chat**, type a question or command in **Message Milo** and press the send icon. Replies appear in the conversation, and follow-up messages can refer to the recent exchange. The transcript follows new replies unless you scroll back. The existing **Robot goal** and **Start LLM control** remain available for goal-driven runs. A chat request supplies the loaded goal only as context, not as a request to start solving it.

Chat accepts one message at a time, up to 2,000 characters, and uses the selected text model, reasoning setting, feedback interval, and turn limit. Each message gets fresh head-camera/sensor feedback and uses the same validated motion tools and stop safeguards. Complete recent model/tool turns are kept in a six-turn context; the visible transcript retains up to 40 entries. A new goal-driven run, starting Voice, episode reset, or server restart replaces this chat context. Chat and Voice do not transfer their conversation histories to each other.

Each sent chat message starts a bounded control run, so the token tracker and exchange feed describe that message's run, while the Chat transcript can span follow-ups. Stop or takeover interrupts the current reply; switching modes does not submit an unfinished message or recording.

## Talk to Milo

Select **Voice** before using the microphone or editing the Realtime connection. Switch back to **Chat** for typed messages and text-model settings.

The **Talk to Milo** microphone targets [GPT Realtime 2 (preview)](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/realtime-2) in Microsoft Foundry, with `reasoning.effort=low`, separate from GPT-5.2 or Luna. In **Realtime connection**, enter its resource endpoint and exact deployment name, choose a voice, and apply the connection. The server keeps authentication credentials out of the browser. Realtime 2 is a public-preview model without a production SLA.

1. Click the microphone button and allow browser microphone access. The first click opens the voice session and begins recording once connected.
2. Speak a command, such as "Move forward a little and tell me when you've stopped". Click the send icon in the same button to finish the utterance. The microphone track is then released.
3. Watch the robot move and hear its reply. User/robot transcripts appear alongside the controls and in the exchange feed. The next microphone click starts another utterance after the reply finishes.
4. Use **End voice**, **Stop**, or **Take manual control** to end voice control and stop playback. Disconnect or episode reset also stops motion and closes voice control. Ending voice brakes before waiting for cloud connection cleanup.

For persistent configuration, add these non-secret settings to `backend/.env` and restart:

```dotenv
FOUNDRY_REALTIME_ENDPOINT=https://your-resource.openai.azure.com
FOUNDRY_REALTIME_DEPLOYMENT=your-gpt-realtime-2-deployment-name
FOUNDRY_REALTIME_VOICE=alloy
```

Use a Foundry resource endpoint, not an `/api/projects/...` endpoint. Deploy the `gpt-realtime-2` model and enter the deployment's actual name, which may differ from the model name. The app leaves that field empty until configured and does not verify model identity from the name alone. Consult the [Foundry Realtime guide](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/realtime-audio-websockets) for deployment and access requirements. Entra authentication uses the existing server-side `DefaultAzureCredential`; `AZURE_OPENAI_API_KEY` is also supported for resource authentication. No Realtime resource is provisioned automatically. UI connection edits last only until server restart.

Voice sessions use a same-origin browser WebSocket to the local backend, which connects to Foundry's GA `/openai/v1/realtime` API. Microphone and reply audio are mono 24 kHz PCM16. AudioWorklet is preferred; a bounded ScriptProcessor fallback supports browsers where worklet startup stalls. Browser microphone access requires localhost or HTTPS. Voice uses the same validated robot tools and observation boundary, with fresh camera/sensors before each model response, including tool-result continuations. The selected feedback interval remains a minimum between updates. Full-duplex barge-in is not implemented; use Stop or End voice to interrupt a reply.

Each recording is capped at 30 seconds; very short recordings are discarded. Sessions default to 30 model responses (200 maximum), allow up to eight tool actions per spoken command, end after 10 minutes at most, and have inactivity/network timeouts. Voice can take over a text-model run; concurrent voice/manual/text-model ownership is rejected. Raw audio is transmitted to Foundry only while recording and is not saved by the local application. Text transcripts, sensor snapshots, and tool results are retained in the bounded session feed. Foundry service retention policies still apply; the Responses adapter's `store=false` setting does not describe Realtime retention.

Voice behavior is verified locally using synthetic audio, scripted Realtime responses, and real robot physics. A live Foundry Realtime deployment and physical microphone/speaker experience remain to be validated after configuration.

Realtime 2 uses the existing `/openai/v1/realtime` transport despite the model's preview status. Session updates include low reasoning and omit the unsupported `truncation` property. Visible commentary and final-answer items are retained together, without exposing private reasoning.

## LLM Control

The LLM control panel selects a model, reasoning effort, robot goal, feedback interval, and turn limit. Built-in profiles are GPT-5.6 Luna (`gpt-5.6-luna`, default when Foundry is configured), GPT-5.4 Nano (`gpt-5.4-nano`), and Gemma 4 E2B through Ollama (`gemma4:e2b-it-qat`). Foundry profiles start with low reasoning; Gemma uses none. Other configured deployments can be selected or added in **Model connection**. Explicit custom profile lists choose their first configured model. Start transfers exclusive control to the model; **Take manual control** cancels inference and brakes before returning control. The global **Stop**, episode reset, and loss of the last operator connection also interrupt model control.

Luna also offers **None** in **Reasoning effort**, sent explicitly as `reasoning: {"effort": "none"}` through the Responses API. [Microsoft documents this option for GPT-5.6 models](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning#tool-calling-with-reasoning-models). It can reduce reasoning work at the expense of planning quality; **Low** remains the app default. Other profiles keep their declared options, and Realtime voice remains on low reasoning. Local scripted tests verify selection and the request payload, not live deployment acceptance or performance.

The model receives the actual head-camera PNG and the `AgentObservation` sensor contract: joints, gripper aperture/contact/load, head angles, odometry, directional bumpers, proximity distances/contact forces, timestamps, episode identifiers, and an optional battery reading for the charging challenge. It never receives spectator poses, scene geometry, or evaluator state. In **Single step**, its eight tools use the existing validated controller: `observe`, `wait`, `drive_base`, `set_head`, `set_arm_joints`, `move_end_effector`, `set_gripper`, and `stop`.

Feedback interval is a minimum wall-clock interval between fresh model inputs, adjustable from 0.25 to 30 seconds (default 2 seconds), including during a run using **Apply rate**. Model requests do not overlap. In single-step mode, physics holds during inference and feedback delays; the explicit `wait` tool advances simulated time while holding position. Navigation plan mode can execute a bounded buffer during inference, as described below. This is inference-based robot control, not model weight training.

### Images and Retained Context

Chat and navigation share two settings, applied on the next start or typed follow-up:

- **Images per request:** 1-8, default 1. Each request includes the current image plus distinct recent model-observation images as they accumulate. The first request may have only one available view. Older images include their own sensor readings and timestamps; they are not current observations. The setting does not trigger extra motion or capture a burst of identical images. All frames come from the robot head camera, never the spectator.
- **Retained context (tokens):** 0-32,768, default 4,096. This is a conservative estimate for past text turns and recent-action summaries, using UTF-8 bytes plus message overhead. Oldest complete turns are discarded without splitting tool calls from replies. Zero removes text history and action summaries; the active request and current feedback remain. Selected images and their paired observations, current goal, tools, and fresh sensors are additional inputs, so this is not a cap on the total provider input tokens.

Historical images are not also replayed through text history. The exchange feed shows every image actually submitted and the retained-context estimate; the usage tracker still shows actual provider-reported tokens. More images increase input cost, and a larger Ollama context may increase GPU/RAM use. These controls do not change Realtime voice conversation retention. No fixed six-turn or three-turn cap remains; a 200-turn memory guard supplements the token budget.

For Luna supervision, start with **2 images and 8,192 retained-history tokens**. This supplies the current view and one distinct historical comparison as available. For longer search/recovery tasks, try 3 images and 16,384 only after checking latency and useful recall. These are starting recommendations, not measured optimal settings; defaults remain 1 image and 4,096. SmolVLA uses its separate single paired head-camera/state input and active skill instruction, not the supervisor's chat-history budget.

System instructions and the active run's goal are resent on every model request, outside trimmed history. Chat also resends the current typed command at every tool step, even with zero retained history. This prevents truncation from deleting the active task, but does not guarantee the model follows it. A new chat follow-up becomes the active command: earlier chat details can be trimmed. Keep task-wide constraints in **Robot goal**, or restate them explicitly when changing the task. Skill instructions remain local task-manager state until cancellation/replacement.

**Request context** shows the latest dispatched request's estimated retained history against its configured budget, retained turns, actual image count and provider-reported input tokens when available. **Active command (resent)** exposes the protected command. The meter is **not** the deployment's full context-window utilization: instructions, tools, current sensors, images and response allowance are additional. No verified full-window limit is configured, so no full-window percentage is invented. Run token totals remain separate. Editing settings affects the next run, not the displayed past request. Pending requests clear their reported input count; disconnected views are marked **Last received**.

### Navigation Plan Mode

Load an apartment scenario, choose your model, select **Navigation plan**, and click **Start LLM control**. Selecting the mode sets the initial feedback interval to 0.25 s; it remains adjustable. **Single step** is the default and remains available for manipulation and comparison. Realtime voice continues using single-step tools.

The model constructs an ordered plan: **Inspect room -> Locate doorway -> Approach -> Cross**, optionally repeating that sequence for another room. Plans can contain up to eight steps, or stop at an earlier stage. The live **Task plan** lists completed, active, pending, failed, and cancelled skills, plus buffered duration and revision. Each completed skill retains the model's visual evidence.

- **Inspect room:** head-only scanning. Completion requires a measured scan of at least 0.4 rad and the model's review of fresh camera feedback.
- **Locate doorway:** head scanning or base rotation, without translation. The same scan requirement applies before the model identifies an opening.
- **Approach / Cross:** bounded base movement, with live clearance checks. Completion requires at least 0.1 m net encoder displacement and no current contact. A crossing stage follows a completed approach. These checks validate motion, not the semantic correctness of the chosen doorway or destination.
- **Motion buffer:** up to four head-only or drive-only segments, no more than two simulated seconds total. The next accepted buffer replaces future motion at a worker slice boundary; it never rewrites elapsed motion. Compatible drive segments retain velocity through acceleration-limited transitions.
- **Refill:** the controller checks remaining motion and wall-clock lease against a latency-adjusted threshold between 0.75 and 1.5 s, keeping one inference request at a time. When model latency cannot fit the window, it plans from rest. Empty or expired buffers brake, discard pending motion, invalidate old revisions, and await fresh feedback without failing the plan. Any now-stale in-flight request is cancelled before replanning. Slow models can still produce pauses; the robot never extends an expired command just to keep moving.
- **Cancellation and faults:** **Cancel navigation plan**, Stop, takeover, reset, mode switching, and final operator disconnect discard pending motion. Lost local feedback, contact, insufficient whole-robot clearance, instability, or skill timeout remain hard safety failures. Inference failure also brakes; worker faults cancel pending inference. Buffer pauses do not extend the skill deadline or restart ended plans.

This mode is a simulator task executor, not a pretrained VLA or automatic Luna/Gemma supervisor. It does not provide a hidden map or a geometric target location to the model. Doorway recognition, room interpretation, and visual completion evidence remain model judgments. The model can replace its plan after reviewing new observations. Autonomous performance and hard real-time behavior are not claimed; see [docs/MECHANICS.md](docs/MECHANICS.md) for freshness and safety bounds.

### Local Navigation Fine-Tune

A separate **native Windows SmolVLA navigation checkpoint** is now trained for short green-bay docking. It uses the head camera plus wheel/head/range readings and predicts forward speed and turn rate. The existing navigation worker executes bounded one-second segments; it does not receive hidden targets from the model. In four fixed held-out cases, the learned controller reached every bay and completed the full parking/stop test in **three**; the fourth kept making corrections and exhausted its request budget.

This is a small docking pilot, not general warehouse navigation. Dataset: 16 scripted and replay-verified routes, 276 paired samples, 12 training episodes and four held out. Fine-tuning: 1,000 actual optimizer updates, separate from the arm checkpoint. Target saturation and the stop deadband are logged; initial inspect/locate scans are scripted. No live UI option or production policy-server change was made. [Training details, limits, evidence and repeat commands](docs/SMOLVLA.md#local-navigation-fine-tune).

An experimental **1,500-total-step stopping candidate** adds 500 updates with training-only balanced stop/move sampling. It completes the previously failed stopping case. In a matched comparison on four new offsets, both checkpoints park successfully, while total commands fall from 81 to 74 but velocity saturation increases from 20 to 27 requests. The original checkpoint stays the default; this is a modest docking improvement, not general navigation. [Results and explicit candidate test command](docs/SMOLVLA.md#navigation-stopping-experiment).

To test the new checkpoint in isolated real physics, using a fresh output directory:

```powershell
./.runtime/env/python.exe -m scripts.navigation_policy --stage evaluate --case 12 --seed 713 --output .runtime/navigation-local-retest
```

**Keep Single step and Navigation plan.** Single step supports precise actions, manipulation, comparison and recovery. Planning is appropriate for multi-stage navigation, with local learned skills potentially executing bounded subgoals. The existing default and both mode controls remain unchanged; this checkpoint is not interchangeable with the seven-axis `Luna + SmolVLA` arm policy.

### Luna + SmolVLA

The opt-in **Luna + SmolVLA** execution mode keeps the selected Foundry model as a task supervisor and runs a local SmolVLA policy through an independent task manager. The supervisor selects a left-arm pick/place instruction, observes progress, and cancels or reports completion; only the worker accepts and executes bounded numeric policy chunks. The base, head and right arm remain fixed. **Local manipulation** shows the policy buffer, latency and rejected chunks, and the **Policy** exchange filter exposes the request/action trace.

This mode requires a **Milo-trained checkpoint** and local policy server, normally at `http://127.0.0.1:8085`. It refuses base-model or incompatible-embodiment execution. Setup, the seven-axis checkpoint contract, cancellation behavior and remaining limitations are in [docs/SMOLVLA.md](docs/SMOLVLA.md). Five scripted, physical-only pick/place demonstrations have been recorded, replayed, and exported with real LeRobot 0.6.1 (3,600 paired frames). Reopened pixels, state/actions, timing, normalization statistics and episode-end action padding are verified. Native Windows CUDA and a 1,100-total-step fine-tuned checkpoint now work. Raw chunks still fail action validation; an explicitly approved, opt-in trajectory adapter produces visible arm/gripper movement. A Windows pacing fix removed premature expiry in the repeated isolated trials: all 70 accepted chunks completed, but none of three trials lifted or placed the cube. The candidate remains disabled for live control. See [timing results and repeat commands](docs/SMOLVLA.md#windows-policy-timing-fix); broader demonstrations, action quality and learned task success remain outstanding.

Policy mode renders timestamped camera snapshots in a separate process instead of rendering continuously on the physics thread. The camera footer reports the actual image size. This avoids that specific blocking path but is not a new hard-real-time performance claim. Single-step, navigation and Realtime voice retain their existing behavior.

### Gemma Through Ollama

Start Ollama and ensure `gemma4:e2b-it-qat` is downloaded (`ollama pull gemma4:e2b-it-qat`). In the app, load a challenge, select **Gemma 4 E2B (Ollama)**, and click **Start LLM control**, or send a message in Chat. **None** reasoning is selected automatically and every Ollama request sets `think=false`. Voice still uses the separately configured Foundry Realtime model.

The default endpoint is `http://127.0.0.1:11434`. **Model connection** allows editing the provider, local endpoint, label, and model tag without overwriting the Foundry endpoint or other profiles. Only HTTP loopback addresses are accepted. `OLLAMA_ENDPOINT` can override the startup endpoint. Gemma works without Azure credentials, and no cloud fallback occurs if Ollama is offline or the model is missing. A configured indicator identifies saved settings, not server reachability or model availability.

Ollama receives only the existing robot camera/sensor history and tool results. Its native tool calls pass through the same single-action validator and worker as Foundry. Requests use at least a 16,384-token runtime context, increased in 8,192-token steps for larger retained-context/image settings (up to 49,152 at the maximum settings), and a 1,024-token output cap. The existing 45-second inference timeout remains. Model support and local memory can still limit accepted input sizes. Private thinking is discarded. The exchange feed and token tracker include local responses; these counts are usage statistics, not cloud charges. Both single-step and navigation plan modes are available. There is no automatic Luna supervision.

Start with short goals and a small turn limit, and keep Stop available. A live integration check completed one head turn followed by Stop, but autonomous navigation and manipulation quality remain unbenchmarked. Small-model direction and planning mistakes are still possible. Cold image/prompt processing can be much slower than repeated requests; a low feedback interval does not guarantee that inference cadence.

### Automatic Idle and Wake

After a single-step task ends, the agent holds position and enters **Agent idle** once the head-camera scene has remained unchanged for five wall-clock seconds. Goal-driven runs also stop when the scenario evaluator reports completion or failure. The model can explicitly report an unachievable task through the `stop` tool. A text-only answer ends the current request; a Voice reply returns to listening-ready before the same idle check. A static image alone does not interrupt unfinished inference or an ongoing task.

While idle, a local camera check runs twice per second without sending model requests, consuming tokens, or advancing physics. Small pixel noise is ignored. A meaningful head-camera change wakes the agent for fresh feedback; spectator orbit, axes, and other operator-only overlays do not. Physics is still step-locked, so there is no simulated movement during idle unless the scene or robot is changed. A new Chat message or explicit push-to-talk interaction wakes normal control immediately.

Text camera wakes preserve the current session, token totals, recent context, remaining turns, and original ten-minute deadline. The model is asked to reassess instead of repeating completed actions. Voice camera wakes assess the view without enabling motion tools or activating the microphone; a new spoken command is required for movement. An open Voice connection keeps its existing connection/inactivity limits. Exhausted turn/time limits disable automatic waking until a new user-requested run.

**Stop**, **Take manual control**, mode switches, reset, last-operator disconnect, and text-model connection changes cancel automatic waking. A camera change cannot override them. Use a new Chat/Voice request or start a new goal run to re-enable the behavior. The status strip distinguishes quiet-camera waiting, idle, and waking; the exchange feed records idle/wake events. An unachievable report is the model's judgment, not proof that no possible solution exists. Network or malformed-response errors remain stopped errors and are not automatically retried.

### Token Usage

The sticky robot status strip shows **Reported tokens** with exact **Total**, **Input**, and **Output** counts for the current text or voice run. Totals accumulate from the provider's usage report after each response, including tool-call continuations. **Idle run** retains the same totals across automatic camera wakes. They stay visible as **Last run** after Stop or takeover and reset when a new user-requested agent run starts or the episode resets. Refreshing the page retains the server's current counts; restarting the server clears them.

On disconnect, **Last received** identifies the retained snapshot. These are reported usage totals, not a live estimate during generation or a billing ledger: interrupted requests or responses without usage data may be missing. Per-response counts remain available in the exchange feed. The tooltip on the tracker summarizes these limits.

### Exchange Feed

The live exchange feed records the information flow in chronological order: session goal/instructions/tools, each submitted camera image and sensor observation, visible LLM responses, requested robot tools, and their actual results. Entries include timestamps, turn numbers, inference latency, token usage, and expandable payloads. Input entries also identify replayed context turns and tool results included in that request. A submitted entry records a request attempt, not proof that the remote model received it; connection failures and interruptions appear as session outcomes.

Filter by Inputs, LLM, Tools, or Session. **Follow latest** tracks incoming events; turn it off or scroll upward to inspect earlier exchanges. Each input thumbnail opens the exact image supplied to that turn, independently of the rotating live camera cache.

The **Copy exchange feed** icon beside **Follow latest** copies all currently loaded, retained events as formatted JSON, including collapsed payloads and events hidden by the active filter. The export records its session/revision and whether earlier events were discarded. Camera URLs are included, but image pixels are not embedded: those links work only while the local session still exists. Clipboard denial is reported and can be retried. Review task text before sharing the export; it can contain your goals and messages.

Text and Ollama requests retain up to six complete model/action turns and prominently repeat a compact summary of the last three actions, including arguments, results, and encoder deltas. **Recent actions supplied** shows that summary in the input feed. It is derived from previous tool results, not spectator or hidden evaluator state. A successful tool call alone is not proof of useful movement.

A meaningful drive that ends in contact with negligible encoder movement reports **NO_PROGRESS**. The model is instructed to look around, choose a safe different motion, or stop instead of repeating the push. Three consecutive drive results with this error stop text/Ollama control without automatic waking. This is a contact-and-encoder heuristic, not a complete stuck detector: wheel slip or lost contact can escape detection, and autonomous recovery remains unverified. It does not implement a task queue or continuous motion.

Only the latest 200 events and their input images are retained in local server memory. A new LLM session, episode reset, or server restart clears the feed. Feed updates are fetched incrementally; detailed sensor data and images are not rebroadcast with every physics state. Private/encrypted reasoning content and authentication credentials are excluded. This is a local exchange viewer, not persistent replay or cloud telemetry.

### Foundry Connection

Backend configuration loads in this order: existing process environment, then `backend/.env`, then root `.env`. Neither file overrides existing variables. Both environment files are ignored by Git. For a project endpoint with Entra authentication:

```dotenv
FOUNDRY_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
FOUNDRY_DEPLOYMENT=gpt-5.6-luna
```

`project_endpoint` and `deployment_name` are accepted aliases, as is `AZURE_AI_MODEL_DEPLOYMENT_NAME`. Without an explicit model list, Luna, Nano, and the separate Ollama Gemma profile are included by default. A legacy single deployment matching one of the Foundry profiles is reused without duplication; a custom Luna deployment name overrides Luna's binding. Other legacy deployments, such as the existing `GPT-5.2`, remain additional options without replacing Luna as the default. A previous real image/sensor Responses probe succeeded against GPT-5.2; Luna and Nano have only been configuration-tested with scripted inference, not invoked live by this update.

For a resource endpoint, use `FOUNDRY_ENDPOINT=https://your-resource.openai.azure.com/`. Resource endpoints can use server-side `AZURE_OPENAI_API_KEY` or Entra; project endpoints use Entra only. `DefaultAzureCredential` uses the operator's existing Azure credentials and requires inference access to the project. Credentials are never entered into the UI. `AZURE_AI_MEMORY_STORE_NAME` is not used by this local controller.

For several persistent model profiles, set `FOUNDRY_MODELS_JSON` to a JSON array containing `id`, `label`, `deployment`, and optionally `provider` (`foundry` by default, or `ollama`) and `reasoning_efforts` (`low`, `medium`, `high` by default; include `none` for compatible non-reasoning profiles). Ollama profiles always use `none`, and their `deployment` is the exact local model tag. This list overrides the built-in profiles and legacy single-deployment aliases. Its first configured profile is the initial selection, or the first profile when none is configured. Connection edits in the UI are server-session only; restart reloads environment configuration.

Configuration status checks for an endpoint and a deployment name; it does not test authentication or whether a deployment exists. `FOUNDRY_PROJECT_ENDPOINT` plus an existing Azure CLI login is supported. Built-in Luna and Nano profiles already have deployment names; edit the profiles if your deployment names differ. Voice uses its own `FOUNDRY_REALTIME_ENDPOINT` and `FOUNDRY_REALTIME_DEPLOYMENT`; an Azure login does not create that deployment. Environment file edits take effect after restarting the backend, and existing process variables take precedence.

The Foundry adapter uses the OpenAI Responses API with `store=false`, preserves complete tool-call/output pairs and encrypted reasoning items in a six-turn rolling context, and caps output at 4096 tokens per request. Defaults are 30 turns (maximum 200), 45 seconds per inference request, and 10 minutes per session. The panel reports actual inference latency and token counts. Malformed, refused, filtered, incomplete, repeated, or multi-action replies cannot bypass the controller's checks. Failed requests stop control; there are no automatic inference retries.

## Current Scope

- Procedural URDF with independently motorized differential wheels, passive low-friction support spheres, head yaw/pitch, two six-joint arms, and parallel-jaw grippers.
- Real gravity, masses, inertia, floor/wheel friction, collisions, fixed-step physics, bounded motor forces, and servo control.
- Base-frame Cartesian IK with endpoint forward-kinematics validation, sampled Cartesian paths, collision preflight, and execution tolerance checks.
- Contact-gated assisted grasps with opposing finger contact, geometric enclosure, aperture/mass/load limits, release and gravity-driven settling.
- Authoritative CPU-rendered head-camera PNGs, RGB/proprioception allowlist serialization, and separate privileged evaluator state.
- Dedicated serialized physics worker, independent stop signal, command envelopes, duplicate-action suppression, stale-observation rejection, and reset/client disposal.
- Manual REST and WebSocket API, original-geometry spectator renderer, head image panel, manual controls, and a session-local action timeline.
- Foundry vision/tool adapter, paced LLM control, configurable model profiles, inference cancellation, manual takeover, and bounded session activity.
- On-demand Foundry Realtime voice control, microphone capture, spoken audio playback, transcripts, and shared robot tool/feedback traces.
- Ten loadable challenge scenes, grouped by capability, with private physics-based checks for ordered navigation, occluded search, cube manipulation, and return-to-charge memory.

See [docs/MECHANICS.md](docs/MECHANICS.md) for frames, controls, and approximations. See [docs/VALIDATION.md](docs/VALIDATION.md) for verified behavior and remaining limitations.

## Milestone Status

1. Mechanical vertical slice: backend mechanics and frontend build/browser gates passed.
2. Foundry control loop and UI: implemented and tested; live GPT-5.2 image/sensor/tool probe passed. Luna binding and autonomous task performance remain unverified.
3. Ollama vision/tool adapter: implemented, with an isolated live Gemma head-turn/Stop check. Other local runtimes and hybrid supervision remain unimplemented.
4. Ten curated challenges and private physics checks: implemented and tested. Randomized challenges, held-out seeds, and persistent benchmark scoring remain pending.
5. Experiment persistence, replay, comparisons, and exports: not implemented.

The robot can execute LLM-selected actions when a compatible Foundry deployment is configured and the user starts control. Autonomous navigation and pickup success have not been benchmarked. There are no saved navigation or pick-and-place demonstration runs yet. Manual and LLM timelines are in-memory and are not persistent replay systems.

## Source Map

- [backend/robot.py](backend/robot.py): shared original robot asset generator and calibration.
- [backend/contracts.py](backend/contracts.py): canonical Pydantic tool registry, command/observation contracts, and interface boundaries.
- [backend/simulation.py](backend/simulation.py): authoritative PyBullet mechanics, private kinematic planning client, sensors, and CPU camera.
- [backend/worker.py](backend/worker.py): serialized simulation thread and interruption handling.
- [backend/challenges.py](backend/challenges.py): predefined tasks, scenes, public goals, and progress criteria.
- [backend/app.py](backend/app.py): manual API, state stream, camera bytes, and static frontend hosting.
- [backend/agent.py](backend/agent.py): Foundry configuration, Responses adapter, and cancellable control loop.
- [backend/realtime.py](backend/realtime.py): Foundry Realtime audio transport and cancellable voice controller.
- [frontend/src/AgentControl.tsx](frontend/src/AgentControl.tsx): LLM selection, connection settings, feedback controls, and activity.
- [frontend/src/ChallengePicker.tsx](frontend/src/ChallengePicker.tsx): task selection, loading, goal preview, and progress checklist.
- [frontend/src/VoiceControl.tsx](frontend/src/VoiceControl.tsx): on-demand microphone, voice connection, and spoken transcripts.
- [frontend/src/main.tsx](frontend/src/main.tsx): operator interface source.
- [tests/test_simulation.py](tests/test_simulation.py): physical command-path acceptance tests.

The selected interpreter is Python 3.11 in `.runtime/env`. Windows installs conda-forge PyBullet 3.25 with OpenBLAS. Linux's proposed dependency pin remains PyPI PyBullet 3.2.7; Linux setup and cross-platform equivalence are unverified. Do not combine benchmarks across these binaries without recording the build difference.