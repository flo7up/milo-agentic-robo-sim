# Validation Status

## Motion Buffer Recovery: 2026-09-11

- Reproduced the reported `BUFFER_EXPIRED` failure with a deterministic clock: rendering/physics slower than wall time could expire a buffer despite fresh local feedback. The public Nano trace also showed a 1.2-second scan with a 1.7-second lease and a preceding 2.797-second response.
- Expiry still brakes, discards pending motion, and invalidates revisions, but now preserves the active skill for fresh-feedback recovery. Refill timing considers wall-clock lease and motion remaining; models whose observed latency cannot fit plan from rest. Revision changes cancel and await outdated requests before new inference. Stop now immediately publishes the cleared buffer.
- All **172 backend tests passed**, including 15 navigation tests, with two existing dependency deprecations. All **29 Edge tests passed** against isolated real physics and scripted inference. Editor diagnostics were clean. New tests cover slow simulation, expiry/exhaustion during inference, no overlapping requests, successful fresh-feedback updates, slow-model fallback, and preserved hard safety stops.
- The user-authorized port-8002 restart restored public provider profiles and a fresh Apartment Search scenario. Page and head camera returned HTTP 200; the controller was idle with no error. Prior public state/trace/camera were preserved under `.runtime/diagnostics/before-buffer-recovery-20260911-115822.*`.
- No live model inference was started for validation. These tests establish controller recovery, not autonomous navigation success or continuous motion with a slow model. Buffer pauses do not extend skill/session deadlines or relax collision, feedback-loss, or Stop checks.

## Reactive Navigation: 2026-09-11

- Implemented opt-in `navigation_plan` execution for both Foundry and Ollama, with ordered inspect/locate/approach/cross skills, model-reviewed completion checkpoints, a two-second replaceable motion buffer, and live plan/buffer/cancellation UI. Single-step remains the default, and Realtime voice retains its original tools.
- Full backend suite: **160 passed, 2 third-party warnings in 159.70 s**. Navigation checks cover a complete four-skill traversal through real doorway geometry, continuous drive velocity across replacement, in-flight worker updates, bounded horizons, scan/movement preconditions, stale/expired revision rejection, floor clearance, buffer and feedback expiry, skill timeout, and inference cancellation after a worker fault.
- API checks exercise Stop, takeover, reset, interaction-mode switching, and last-operator disconnect during buffered motion. Pending motion is discarded, existing completions remain distinguishable from cancelled steps, and manual placement is rejected during navigation ownership. Both model adapters are tested with their actual mode-specific request schemas.
- Production build and editor/Pylance diagnostics pass. Full Edge suite: **29 passed in approximately 2.8 minutes**. The navigation fixture verifies skill progression, motion while inference is pending, canvas pixels on desktop/mobile, buffer cancellation, and reset back to single-step mode. Desktop/mobile screenshots were inspected. The existing bundle-size warning remains.
- These are scripted-policy and real-physics checks, not live model recognition or autonomous navigation results. The model selects doorway semantics and supplies visual evidence; mechanical checkpoints do not verify that claim. Slow inference can exhaust a buffer and pause the robot. Constant-velocity lookahead and software watchdogs are conservative simulator safeguards, not certified real-hardware safety or a general motion planner. No cloud resource or model deployment was created.

## Feed Export and Stalled Driving: 2026-09-11

- The inspected live problem trace already replayed six action/result pairs. Later forward commands requested 0.1 m/s for 1 s, reported `ok`, and produced only about 0.000007-0.000052 m encoder displacement with front contacts around 64-67 N. Reported input sizes were about 13.4k tokens, below the configured 16,384-token context; this does not independently prove that the provider used every history item. The public diagnostic trace was preserved at `.runtime/diagnostics/repeated-forward-feed.json`.
- Contact-backed stalled drives now return `NO_PROGRESS` with inspection/replanning guidance. Real apartment physics verifies the error, normal head inspection, zero-motion commands, and retreat. Scripted model tests verify the last-three-action summary reaches the actual request and trace once, and three consecutive stalled drives stop without automatic wake. No hidden object/room geometry is added to feedback.
- **Copy exchange feed** exports all loaded retained events, irrespective of filters or collapsed payloads, with revision/truncation metadata and session-local camera URLs (not embedded images). The Edge clipboard test uses a clipboard shim to verify full payload equality, failure/retry, disabled empty state, reset, and mobile layout. It does not independently validate OS clipboard permission policy on other browsers.
- Full backend suite: **141 passed, 2 third-party deprecation warnings in 222.74 s**. An initial run exposed an existing completion-publication race; publishing `busy=false` atomically with the final observation fixed the isolated failing check and the full rerun.
- Frontend production build, editor diagnostics, and full Edge suite passed: **28 tests in approximately 2.8 minutes**, with real physics and scripted inference. The existing bundle-size warning remains. The mobile copy-control capture was inspected.
- These results establish export correctness, clearer feedback, and bounded repeated pushing. No new live model run or autonomous recovery/navigation success is claimed.

## Ollama Integration: 2026-09-11

- Gemma 4 E2B (`gemma4:e2b-it-qat`) is selectable alongside Luna, Nano, and legacy Foundry profiles. Ollama uses an independently validated HTTP loopback endpoint and `think=false`; local sessions never construct the Foundry adapter or forward cloud credentials.
- Full backend suite: **138 passed, 2 third-party warnings in 121.64 s**. Added native image/tool conversion, paired tool-result history, private-reasoning exclusion, incomplete/invalid/multiple-call rejection, missing-model/offline messages, actual worker movement, and pending-HTTP cancellation checks.
- Follow-up adapter checks: **11 passed in 6.51 s**, including preservation of native Ollama call IDs and a real-physics check that a repeated ID cannot repeat motion.
- Production build and editor diagnostics pass. Full Edge suite: **27 passed in approximately 2.6 minutes** with scripted inference and real physics. It verifies local configuration edits, None reasoning, Stop, endpoint independence, preservation of Foundry choices, and use without a cloud endpoint. Desktop/mobile full-page captures were inspected. The existing bundle-size warning remains.
- A separate **live Ollama** test used fresh kitchen camera/sensor observations and the production controller in an isolated physics instance. Gemma called `set_head` with yaw -0.3, pitch 0.2, duration 1 s, then `stop`; both returned `ok`, the measured head reached the target, and the camera changed. Inference took **12.469 s** then **0.782 s**, with 10,043 input and 46 output tokens reported across two requests. The run did not contact Foundry or alter the operator episode. Local report: `.runtime/ollama-app-check.json`.
- This verifies integration and one explicit head-control task, not autonomous room navigation, grasping, continuous control, or hybrid Luna supervision. Earlier simple dry runs showed direction mistakes despite schema-valid calls; retain conservative goals and operator Stop.

## Recovery Check: 2026-09-11

Recovered the compatible API, physics worker, sensor contracts, robot geometry, frontend, and regression tests from VS Code local history after earlier mechanical-only files replaced them. Each replaced file was backed up under `.runtime/recovery-20260911/` and both backup and restored contents were hash-checked. Credential files and the surviving Foundry model configuration were not changed.

- Full backend suite: **121 passed, 2 third-party deprecation warnings in 119.61 s**.
- Frontend production build and dependency consistency check passed with Vite 7.3.6 and Playwright 1.55.1. The existing bundle-size warning remains.
- Full Edge suite: **25 passed in approximately 2.3 minutes**, using real physics and scripted inference on isolated port 8001. One Windows socket-teardown connection-reset callback was logged; all assertions passed.
- Desktop/mobile apartment screenshots and canvas-pixel checks verified the restored 3D view, head camera, room fixtures, sensors, and outcome feedback.
- Luna, Nano, and configured Foundry alternatives remain available. No local SLM is wired into the app or invoked by this recovery. The standalone local-model probe remains separate and unchanged.
- No live model invocation or autonomous task-performance claim is part of this recovery. The sections below retain their original historical validation details.

Date: 2026-09-10 (historical results below). Mechanical, LLM, Realtime voice, and six predefined challenge slices have passing local tests. A live GPT-5.2 image/sensor/tool probe passed; GPT-5.6 Luna, GPT-5.4 Nano, and live Foundry Realtime deployment access, physical microphone quality, and autonomous task results remain unverified. Randomized challenge benchmarks and experiment persistence remain unimplemented; Ollama was added in the integration check above.

## Verified with Real Physics

The Python tests run against conda-forge PyBullet 3.25, Python 3.11.16, NumPy 1.26.4 and SciPy 1.17.1 on Windows. They do not substitute mocks for mechanics.

- Both six-joint arms reach the curated floor target through `move_end_effector`.
- Both arms have fixed shoulder brackets bridging the torso and the unchanged shoulder pivots. Asset tests verify visual/collision overlap at the torso and pivot; desktop/mobile captures show the connected assembly.
- Both arms close on actual opposing finger contacts, establish an assisted grasp, lift the cube, release it, and allow it to settle under gravity.
- Closing far from a cube does not establish a constraint.
- Unreachable IK targets fail without advancing physics.
- Collision-rejected arm targets do not move.
- Differential wheel driving translates and rotates the base; a wall blocks translation.
- Head yaw/pitch change the authoritative frame and achieve the tested target angles.
- Agent observation top-level serialization excludes evaluator bodies, objects, depth, goals and exact robot position.
- Stale observation sequences, duplicate action IDs, stopped runs, and obsolete episode commands do not cause unintended extra motion.
- A stop event interrupts a two-second worker command before completion.
- Reset replaces the run ID/epoch, disposes the previous clients, and rejects obsolete live-frame access.
- Manual HTTP endpoints return real camera PNGs and route motion through the validated simulator.
- Cross-origin mutation requests are rejected.

The full backend suite, `./.runtime/env/python.exe -m pytest -q`, completed with **117 passed, 2 warnings in 173.53 s**, including Kitchen to Bathroom, Luna/Nano configuration, textures, proximity sensors, structured outcomes, idle/wake lifecycle, Chat/Voice modes and follow-ups, Apartment Search, manual placement, in-flight live previews, return-to-charge energy/memory, predefined challenges, shoulder attachment, LLM controller, API, exchange-feed, and Realtime voice tests. Before LLM integration, the mechanical suite had 12 passing tests. Test-suite wall times are not simulator FPS measurements or model performance claims. CPU camera throughput has not been profiled.

The root launcher `./start.ps1 -SkipBuild -Port 8000` was run successfully after building the frontend. It serves the operator interface at http://127.0.0.1:8000 and API documentation at http://127.0.0.1:8000/docs.

The HTTP test emits third-party Starlette/httpx and AnyIO deprecation warnings. It passes; these dependencies should be reviewed before a release.

## Verified in Microsoft Edge

- `npm --prefix frontend run build` passes with TypeScript 5.9.2 and Vite 7.3.6.
- `npm.cmd --prefix frontend test -- --reporter=dot` passes with Playwright 1.55.1: **25 passed in approximately 2.9 minutes** against an isolated PyBullet backend on port 8001 with test-only scripted inference, including the sixth scenario and Luna/Nano controls. The Windows test server logged two connection-reset callbacks during socket teardown; all assertions passed.
- Desktop (1440 x 1000): nonblank spectator canvas, visible robot, orbit-driven pixel changes, and unchanged authoritative head-camera reference during spectator orbit.
- Manual forward driving changes the authoritative frame and increases measured forward odometry by more than 0.1 m; head commands, stop, and resume update the interface and timeline.
- Mobile (390 x 844): no horizontal page overflow with arm/gripper controls open, nonblank spectator pixels, and a decoded 640-pixel-wide head-camera image.
- Tests reset the episode independently. Full-page desktop/mobile screenshots were captured and visually inspected.
- LLM controls: default Luna/low/2-second settings, disabled unconfigured start, connection editing, adding/selecting deployments, goal submission, real scripted-tool motion, live rate updates, manual takeover, and Stop.
- The LLM connection panel and controls fit the 390-pixel viewport. Dedicated desktop/mobile screenshots were inspected. Browser tests never send inference requests to the user's Foundry project.
- The sticky activity strip distinguishes executing commands from model thinking and feedback waits in real-physics manual/scripted-LLM tests. A separate UI-state fixture covers stopped/stopping, completed, error, voice, disconnect, and reconnect precedence. Reopening the WebSocket alone cannot restore a running label before fresh state arrives. The clock is outside the announcement region, and reduced-motion preferences disable animation. Desktop, 390-pixel, and 320-pixel screenshots were inspected; the strip stays visible while scrolling without horizontal overflow.

The browser stop check verifies the stopped/resumed UI states. In-flight interruption is covered separately by worker and agent tests. Vite reports a non-blocking bundle-size warning (approximately 760 kB minified / 207 kB gzip); code splitting has not been tuned.

## Materials, Sensors, Outcomes

- Six generated bitmap textures have verified dimensions and pixel variation. Removing textures changes the authoritative camera pixels while leaving physics snapshots unchanged. API tests verify shared material delivery and reject unknown asset names. Edge confirms all six materials load, renders nonblank desktop/mobile scenes, and reports no page errors. Textured apartment screenshots were inspected.
- Real ray tests verify the expected distance to a known wall, correct side readings after base rotation, null clear readings, and no object identity/pose leakage. Real wheel motion produces directional contact forces; normal floor contact does not trigger bumpers. Browser readings update during motion independently of model feedback.
- Structured-outcome tests distinguish agent-reported completion/unachievable from physics verification and retain reasons through interruption. Real scenario browser tests show verified completion and operator-assistance labels. Controlled UI tests cover failures, errors, limits, interruptions, normal replies, blocked actions, newer-action priority, reset clearing, and disconnect overriding stale success.
- Result feedback remains visible while scrolling and during idle; 1440-, 390-, and 320-pixel captures were inspected. All validation uses real local physics, scripted inference, and synthetic voice input. No live model success or real-hardware sensor/safety guarantee is claimed. A shared-terminal interruption during the first full browser run was resolved by rerunning the entire suite in isolation.

## Idle and Wake Evidence

- Camera detector tests verify the five-second stability boundary, small-noise tolerance, meaningful-change revision, and real head-camera changes without stepping physics or incrementing model observation sequences.
- A real five-second controller test verifies no new inference or observation during idle, then one camera-triggered wake with fresh input and preserved session usage. Additional tests cover structured unachievable outcomes, real parking completion stopping continuation, pending static-camera inference remaining active, turn-limit enforcement, and stale stop-revision refusal.
- API tests verify Stop, takeover, mode switch, reset, and final disconnect cancel the watcher and clear automatic waking. Voice tests cover camera wake with `tool_choice=none`, no automatic recording, new push-to-talk wake, and watcher cleanup on halt.
- Edge tests verify visible idle status, stable tokens, unchanged observation sequence while idle, spectator axes not waking the model, head-camera motion waking the same text session, Chat interaction wake, and Voice idle with microphone off. Desktop/mobile idle screenshots were inspected. All inference is scripted, all robot motion is real local physics, and microphone input is synthetic.
- The feature intentionally requires a finished/unachievable task plus five quiet seconds. It does not infer task impossibility from stillness or cancel a slow model request. The detector's tolerance and 0.5-second sampling are heuristics; tiny or shorter-than-sample scene changes can be missed. Idle monitoring does not make physics free-running, open the microphone, reset request budgets, or establish autonomous model performance.

## Chat and Voice Evidence

- Controller tests exercise a no-motion text answer followed by a movement request and tool-result continuation. Follow-ups retain prior text and paired robot feedback in bounded history; model inputs exclude spectator/evaluator state. Empty messages, concurrent requests, and stale conversation IDs are rejected.
- API tests verify Chat defaults, explicit mode selection, operator-connection and same-origin guards, text/voice exclusivity, cancellation of pending chat on switching, and reset invalidation. A late voice start cannot activate while Chat is selected.
- Edge tests send typed questions and follow-up movement commands through real physics, verify user/assistant transcripts and the latest reply's visibility, and switch away from an intentionally pending response. Keyboard tab switching retains focus. Voice tests explicitly choose Voice and verify switching to Chat while recording releases microphone tracks and ends server ownership. Mode selection alone does not request microphone access.
- Desktop/mobile Chat and Voice screenshots were inspected. Existing goal runs, voice replies, token counters, challenges, live views, and manual controls pass the full browser suite. Tests use scripted inference and synthetic microphone input, not a new live model conversation. Chat and Voice do not share their model context; transcript persistence beyond the current controller/episode is not implemented.

## Token Tracker Evidence

- The sticky tracker uses the existing server totals, showing input, output, and their sum for a text or voice session. Text fixture checks verify 10 input + 5 output = 15, unchanged while another response is pending and retained after takeover. The two-response voice fixture verifies 20 + 10 = 30, retained after End voice and reset on the next voice run. Episode reset clears the display.
- A controlled WebSocket fixture verifies exact digit grouping, current/last/no-run scope, large counts, and retention marked `Last received` after disconnect until fresh state arrives. The tracker is outside the robot-status live region. Desktop, 390-pixel, and 320-pixel screenshots were inspected; counts fit and remain visible while scrolling.
- This is frontend tracking of provider-reported response usage. Missing usage or interrupted requests are not estimated; the UI tooltip and README disclose that these totals are not a billing ledger. Each Chat message starts a bounded control run and resets these counters, while the conversation transcript retains follow-ups. No paid model request was made for tracker or mode validation.

## Manual Placement Evidence

- Worker tests verify a new physical position and camera image, preserved heading/height, unchanged simulation time/odometry, one fresh observation, and an incremented manual-placement count. Stale, out-of-floor, obstacle-overlapping, and stopped requests leave the authoritative pose unchanged. Stop cancels a placement already queued behind another worker operation.
- Existing real-grasp tests for both arms reject repositioning while carrying an assisted cube, retaining the original pose, observation sequence, and grasp.
- API checks reject nonfinite coordinates, wrong dimensions, extra fields, cross-origin requests, stale episodes, and both text/voice ownership conflicts. Reset clears placement history. No placement tool is exposed to models.
- Desktop mouse and mobile native-touch tests select the rendered robot and drag it. Before release, the scene state and camera remain unchanged; after release, the real pose changes, the camera refreshes, and the footer shows manual intervention. Keyboard tests cover preview cancellation, blocked-drop rollback, successful placement, Stop, and reset. Selection/preview and applied-placement screenshots were inspected.
- These are operator-editing checks, not autonomous navigation results. Background orbiting and existing live camera/model/voice workflows still pass.

## Live Preview Evidence

- A wall-paced two-second head movement produces multiple distinct PNGs and changing pose snapshots before completion. Intermediate previews do not increment the model observation sequence or enter its frame cache; the final preview reuses the final observation image. The preview cache is bounded and cleared on close.
- API checks verify initial/final PNG delivery, `no-store`, preservation of model-observation images, and rejection of old-run or missing preview URLs after reset.
- Desktop and mobile Edge tests execute a two-second wheel movement requested by the scripted LLM. At least three decoded camera frames appear while the controller is still acting, with changing head-image and spectator-canvas pixels and increasing simulated timestamps. Model observation sequence remains constant during motion and only one model turn has occurred.
- The mobile test delays camera responses by 180 ms, verifies at most one preview request in flight, and still observes motion before completion. Both tests receive the final frame and a fresh zero-time image after reset. In-flight desktop/mobile screenshots were inspected.
- These are real-physics/scripted-inference checks, not a new live Foundry run or a measured 10-fps guarantee. Model feedback intervals are unchanged; physics still holds between commands.

## Predefined Challenge Evidence

- Kitchen to Bathroom: a real-wheel route leaves the kitchen through its doorway, crosses the hall, enters the bathroom, and stops with over 4 m of measured travel. The initial pose is in the kitchen without collision and with zero odometry. Rays verify the dividing wall blocks a direct route and both doorways are open. Arrival checks reject the kitchen, hall, partial doorway occupancy, motion, and lack of floor contact; leaving the bathroom clears completion. Camera bytes change, and the upper toilet fixture is visible from the tested arrival pose.
- The sixth preset's browser check verifies loading, the room-identification goal, 100-turn default, arrival feedback, and reset to the kitchen. Its completion-display step uses recorded manual placement plus a physical wait; wheel-route feasibility is tested separately. Inspected captures under `frontend/test-results/` include `kitchen-start-camera.png` (fridge, sink, oven), `bathroom-arrival-camera.png` (toilet, bathtub, basin), and `kitchen-bathroom-desktop.png` / `kitchen-bathroom-mobile.png`. Both viewports have nonblank spectator pixels and mobile has no horizontal overflow. Text/voice tests exclude private initial pose, fixture names, room labels, arrival geometry, and progress. No live model room-recognition test or semantic grading is claimed.
- Apartment Search: real-wheel commands travel through both doorways, reject inspection of the red distractor, and complete a stationary inspection of the yellow cube. Pixel checks confirm the actual head image contains the yellow target. Physics rays verify initial wall occlusion and reject a nearby viewpoint behind the outer wall. Scoring checks reject excessive distance, occlusion, base/head motion, lack of grounding, and repeated same-time observations; looking away resets dwell. Backend-only target metadata stays out of the public preset and observations.
- Apartment browser checks load/reset the fifth preset, autofill its goal and 100-turn limit, and render nonblank desktop/mobile views. The mobile completion-display test uses an explicitly recorded manual placement followed by a physical wait; route feasibility is covered separately by wheel commands. Apartment and target-camera screenshots were inspected.
- Park in the Bay: real wheel commands move the base and wheels completely inside the green zone, then brake to a measured rest on the floor. The browser displays completion and clears it on reset.
- Tidy the Cube: real Cartesian/gripper commands lift the red cube, carry it to the blue floor zone, and release it. Both left- and right-arm placements are exercised in the sorting scene.
- Color Sort: a scripted sequence completes both cubes within a single physics episode; the first placement remains correct after the second.
- Remember and Recharge: a scripted wheel-control route goes around the screen, completes the survey, receives a low-battery sensor reading, returns to the original charger, and reaches at least 90% via bounded waits. A physical ray check confirms direct charger visibility is obstructed. Zero charge interrupts/rejects motion; reset restores full battery and all mission stages.
- Charging tests reject initial camping, charging while moving, partial dock occupancy, and the wrong zone. Repeated observations do not charge the battery. The wait tool is duration-bounded, deduplicated, and respects Stop.
- The text-model memory test verifies the first actual image/sensor observation remains in the eighth request after rolling context eviction. Neither charger coordinates nor evaluator state is added; current feedback remains distinct and last. Battery payloads contain only percentage, low, and charging fields.
- Evaluator tests reject pushing without a recorded lift, hovering, partial containment, nonzero motion above settling thresholds, and swapped colors. Success requires actual floor contact; conservative collision AABB height alone is not used to infer grounding.
- Valid lift evidence includes opposing physical finger contacts after the optional grasp-assist constraint releases. Score checks do not depend solely on that approximation.
- API tests verify six public options, scene/goal/progress replacement, reset of the selected challenge, model-configuration preservation, same-origin mutation protection, and rejection of stale commands after changing tasks during pending inference.
- Voice and text-model boundary tests confirm public goals are available without leaking scene object names, coordinates, or evaluator progress into observations/session instructions.
- Browser tests load each distinct scene, verify goal autofill, checklist counts, head image/canvas output, reset behavior, and mobile fit. Desktop/mobile screenshots were inspected; parking uses a wider initial spectator view.
- These are curated, repeatable scenes with physics-based live checks, not an autonomous-model benchmark. No paid model run was made to establish challenge success. Charging has a timed survey stage and apartment search has a timed visual inspection; generalized dwell scoring, randomized seeds, spherical objects, and persisted results remain out of scope.

## Realtime Voice Evidence

- The voice target is now `gpt-realtime-2` (public preview) with `reasoning.effort=low`. Tests verify the outgoing session update omits unsupported `truncation`, public configuration names the intended model without inventing a deployment, and commentary/final-answer messages are retained without private reasoning. The focused Realtime/API suite passed 20 tests; both voice browser tests and the production build passed after this update. This is protocol/fixture validation, not a live Realtime 2 model result.
- The Foundry GA WebSocket adapter is validated against a real local WebSocket server for JSON send/receive. API-key headers stay server-side; request URLs use the GA resource endpoint and deployment query parameter. Entra uses `https://ai.azure.com/.default`.
- Scripted Realtime events plus real PyBullet verify microphone packets, manual commit, camera/sensor injection, validated wheel movement, tool-result continuation, output PCM audio, and user/robot transcript events.
- Invalid, unknown, multiple, or refused tool requests cannot cause motion. After the model calls stop, a subsequent unsolicited tool call is rejected even if the model ignores `tool_choice=none`.
- Explicit microphone activation is required before audio packets are accepted. Pending inference is canceled on disconnect, and disconnect during motion brakes before a deliberately delayed cloud socket close completes. Global Stop releases exclusive voice ownership.
- Edge browser tests use an oscillator-backed synthetic MediaStream, never the user's microphone. They verify no microphone request before clicking, capture to PCM, real scripted-tool motion, scheduled audio playback, track release after sending, End voice, global Stop while recording, and denied-permission cleanup.
- Desktop/mobile voice screenshots were inspected. Captions, connection inputs, microphone/send buttons, and End voice fit at 1440 and 390 pixels. The audio worklet did not initialize within the bound on this machine; the native ScriptProcessor fallback passed the end-to-end test. Real physical audio quality and the preferred worklet path still need device validation.
- WebSockets 17.1 was already installed and locked. The project now declares it directly, and `uv lock --offline` succeeds. Downloads for an older optional OpenAI Realtime extra failed TLS, so the implementation uses the installed WebSockets library with Foundry's documented GA protocol. No certificate verification or registry policy was bypassed.
- No live Realtime endpoint, speech-recognition accuracy, conversational latency, voice quality, or speech-understanding model result is claimed. No paid Realtime deployment was created or invoked.

## Exchange Feed Evidence

- Recorded sensor JSON and retained PNG bytes match the exact inputs passed to the model adapter. Visible response text, requested arguments, and tool results are recorded in execution order, including results of interrupted motion.
- Pending inference exposes its submitted input before an output exists; cancellation creates an interruption outcome without fabricating an LLM reply.
- Private/encrypted reasoning and credential markers are excluded from feed payloads. Malformed nonfinite arguments remain safely serialized text in the visible model output.
- Incremental trace cursors and the 200-event retention bound are tested. Input images remain available after 17 later observations evict them from the simulator's observation-frame cache; episode reset invalidates old trace image URLs. Live operator previews have a separate cache.
- Browser tests verify chronological order, decoded input images, sensor/result expansion, input/LLM/tool filters, follow-latest scrolling, inspection without autoscrolling, and reset clearing. A populated mobile feed fits the viewport; desktop/mobile captures were inspected.
- These feed tests use scripted inference with real physics. No new live model request or cloud telemetry export was made for this feature. Feed data is local and session-scoped, not persisted to disk.

## LLM Control Evidence

- Built-in profiles now bind Luna to `gpt-5.6-luna` (default) and Nano to `gpt-5.4-nano`. The legacy GPT-5.2 setting remains an additional profile. Eight focused profile tests and two API checks passed for alias handling, duplicate prevention, custom JSON precedence, actual adapter selection, and reset persistence. Three browser checks passed for Luna's initial selection, Nano's scripted run, and existing custom-profile controls. Build and editor diagnostics pass. The full-suite results above include these configuration changes.
- Custom-list regression: an explicit list with unconfigured Luna and configured GPT-5.2 still selects its first configured profile. Browser tests retain precise missing-endpoint versus missing-deployment messages without falsely reporting an authentication failure. Built-in bindings and the configured indicator do not establish live deployment access; no new Luna or Nano inference or cloud deployment was performed.
- Scripted model responses drive real physics through the canonical tool schema and existing worker; no mock replaces robot mechanics.
- Tests cover feedback pacing, updates during a pending wait, image/sensor allowlisting, tool-call/output continuation, malformed and nonfinite arguments, unknown tools, multiple actions, duplicate IDs, refusals/filtering, inference timeout, repeated stop during motion, and client cleanup.
- HTTP tests cover configuration validation, model selection, manual/LLM ownership, reset invalidation, last-operator disconnect, and atomic publication of ready workers during resets.
- OpenAI SDK requests were intercepted in a test to verify the actual payload includes the chosen deployment, `reasoning.effort=low`, image input, eight function tools, no parallel tool calls, and `store=false`.
- `backend/.env` is loaded ahead of root `.env` without overriding process values. The supplied project endpoint and `deployment_name` aliases are supported. Tests isolate themselves from the user's connection and credentials.
- A real, no-motion probe used the configured `GPT-5.2` deployment with a live robot image and sensor payload: response `completed`, tool `observe`, 2197 input tokens and 27 output tokens. No tool from this response was executed. This verifies the supplied connection, image input, low reasoning, and function calling; it does not establish task performance or Luna availability.
- Project-level standard `/models` listing returned 404, and the Azure resource lookup did not find the account in its current context. A successful Responses call is the connection evidence; no cloud resource was created or changed.

## Installation Findings

- The initial Windows PyPI source-build route was unusable without a C++ build toolchain.
- Windows binary installation succeeded through conda-forge after selecting OpenBLAS. This avoided the failing MKL/ICU/XML dependency chain.
- Launch the environment's Python executable directly. A fresh shell's `micromamba run` silently failed on this machine, although the installed interpreter and native physics imports worked.
- Node.js 24.12.0 and npm 11.6.2 are installed.
- Earlier requests to `https://packagefeedproxy.microsoft.io/npm/` timed out, and offline caches lacked required packages. Access is now restored; installation through that approved feed succeeded.
- The generated frontend lockfile contains 133 HTTPS tarball URLs on `ms-feed-25.pkgs.visualstudio.com`. A fresh metadata request through the CFS proxy confirmed that internal tarball host. No blocked public registry was used.
- Updated Vite from 7.1.3 to 7.3.6 and Playwright from 1.55.0 to 1.55.1 to resolve the reported dependency advisories. The subsequent install audit reports **0 vulnerabilities**.

## Not Verified or Not Implemented

- Browser-driven end-to-end arm pickup/release, additional viewport sizes, browsers other than Edge, and prolonged disconnected/reconnected sessions are not yet covered by the frontend suite. Both-arm pickup/release is covered by real backend physics tests.
- A configured speech-to-speech Realtime deployment is required for real voice use. Full-duplex interruption/barge-in, physical microphone/speaker quality, and continuously advancing physics during model thinking are not implemented or verified by synthetic voice tests.
- Luna inference and full autonomous navigation/pickup/release runs remain unverified. Ollama is implemented as described above; other local OpenAI-compatible and Foundry Local adapters are not implemented. Scripted inference exists only in tests, not as a selectable production model.
- JSON argument repair, continuous real-time physics during inference, automated model-capability discovery, persistent memory, cost estimates, and prompt-injection robustness evaluation are not implemented. The instruction boundary and tool validation exist, but are not a robustness guarantee.
- The full benchmark challenge specification, generalized timed dwell scoring, held-out/randomized seeds, and challenge import/export are not implemented. Six curated presets have private manipulation, navigation, staged energy, and visual-search checks. Autonomous memory, apartment-search, and kitchen/bathroom recognition performance remain unmeasured; room descriptions are not automatically graded.
- SQLite/local artifact recording, OpenTelemetry, retention limits, replay, result comparisons, seed batches, downloadable results, and saved end-to-end demonstration runs: not implemented.
- Linux startup, optional Azure deployment, authenticated production serving, and cross-platform reproducibility: not verified.

## Resume Point

The frontend installation blocker is resolved. Reproduce the mechanical bench gates with:

```powershell
npm --prefix frontend ci --registry=https://packagefeedproxy.microsoft.io/npm/
npm --prefix frontend run build
./.runtime/env/python.exe -m pytest -q
./start.ps1 -SkipBuild
```

Run `npm --prefix frontend test` with port 8001 free; it manages its own isolated test server. The local UI on port 8000 defaults to `gpt-5.6-luna` and offers `gpt-5.4-nano` for testing, retaining legacy deployments such as GPT-5.2 as alternatives. Voice requires `FOUNDRY_REALTIME_ENDPOINT` and `FOUNDRY_REALTIME_DEPLOYMENT`, or equivalent UI connection settings. Remaining gates include a live Realtime conversation, verified Luna/Nano connections, and an observed autonomous task run. Preserve the robot-observation boundary; do not report autonomous results until they have actually been executed and recorded.