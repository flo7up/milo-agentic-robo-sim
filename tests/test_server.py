import asyncio
from types import SimpleNamespace

import pytest
import uvicorn

from backend.server import IdleServer, ServerActivity, lab_is_busy


def test_unused_timeout_starts_after_startup_and_has_explicit_opt_out():
    now = [0.]
    activity = ServerActivity(None, idle_timeout_s=300., clock=lambda: now[0])
    now[0] = 1000.
    assert not activity.expired()
    now[0] = 1299.
    assert not activity.expired()
    now[0] = 1300.
    assert activity.expired()
    assert not activity.expired(busy=True)
    now[0] = 1599.
    assert not activity.expired()
    assert not ServerActivity(None, 0., clock=lambda: now[0]).expired()
    for invalid in (-1., float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ServerActivity(None, invalid)


@pytest.mark.parametrize("kind", ["http", "websocket"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_connected_clients_and_recent_requests_prevent_shutdown(kind, cancelled):
    now = [0.]
    entered, release = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        entered.set()
        await release.wait()

    activity = ServerActivity(app, 300., clock=lambda: now[0])
    request = asyncio.create_task(activity({"type": kind}, None, None))
    await entered.wait()
    now[0] = 500.
    assert activity.connections == 1 and not activity.expired()
    if cancelled:
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    else:
        release.set()
        await request
    assert activity.connections == 0
    now[0] = 799.
    assert not activity.expired()
    now[0] = 800.
    assert activity.expired()


async def test_lifespan_does_not_count_as_a_connected_client():
    calls = []

    async def app(scope, receive, send):
        calls.append(scope["type"])

    activity = ServerActivity(app)
    await activity({"type": "lifespan"}, None, None)
    assert calls == ["lifespan"] and activity.connections == 0


@pytest.mark.parametrize("owner", ["connections", "lock", "agent", "recording", "home_recording", "command", "home", "skill", "continuous", "ros", "recorder", "power"])
def test_every_active_owner_blocks_unused_shutdown(owner):
    worker = SimpleNamespace(latest={"busy": False}, power_transition=False, home_mission=None,
        skill=None, continuous=None, ros_navigation=None, recorder=None)
    lab = SimpleNamespace(worker=worker, connections=0, lock=asyncio.Lock(),
        agent=SimpleNamespace(active=False, recording_active=False), home_recording=None)
    assert not lab_is_busy(lab)
    if owner == "connections":
        lab.connections = 1
    elif owner == "lock":
        lab.lock = SimpleNamespace(locked=lambda: True)
    elif owner == "agent":
        lab.agent.active = True
    elif owner == "recording":
        lab.agent.recording_active = True
    elif owner == "home_recording":
        lab.home_recording = object()
    elif owner == "command":
        worker.latest["busy"] = True
    elif owner == "recorder":
        worker.recorder = object()
    elif owner == "power":
        worker.power_transition = True
    else:
        setattr(worker, {"home": "home_mission", "ros": "ros_navigation"}.get(owner, owner), SimpleNamespace(active=True))
    assert lab_is_busy(lab)


async def test_idle_server_requests_graceful_shutdown_and_honors_explicit_stop(monkeypatch):
    now = [0.]
    busy = [False]
    activity = ServerActivity(None, 5., clock=lambda: now[0])
    server = IdleServer(uvicorn.Config(activity), activity, lambda: busy[0])

    async def base_tick(self, counter):
        return self.should_exit

    monkeypatch.setattr(uvicorn.Server, "on_tick", base_tick)
    assert not await server.on_tick(0)
    now[0] = 6.
    busy[0] = True
    assert not await server.on_tick(1)
    busy[0] = False
    now[0] = 11.
    assert await server.on_tick(2)
    busy[0] = True
    server.should_exit = True
    assert await server.on_tick(3)


def test_finished_recording_does_not_keep_an_unused_server_alive():
    lab = SimpleNamespace(worker=None, connections=0, lock=asyncio.Lock(),
        agent=SimpleNamespace(active=False, recording_active=False), home_recording=SimpleNamespace(finished=True))
    assert not lab_is_busy(lab)


@pytest.mark.parametrize("interruption", ["request", "work", "archive_error"])
async def test_activity_during_preservation_cancels_shutdown(monkeypatch, interruption):
    now = [0.]
    busy = [False]
    activity = ServerActivity(None, 5., clock=lambda: now[0])
    activity.expired()
    now[0] = 6.

    async def preserve():
        if interruption == "request":
            activity.last_activity = 6.
        elif interruption == "work":
            busy[0] = True
        else:
            raise OSError("Archive unavailable")

    async def base_tick(self, counter):
        return False

    monkeypatch.setattr(uvicorn.Server, "on_tick", base_tick)
    server = IdleServer(uvicorn.Config(activity), activity, lambda: busy[0], preserve)
    assert not await server.on_tick(0)


async def test_idle_snapshot_preserves_unsaved_map_without_writing_map_store(tmp_path):
    import json
    from backend.server import preserve_idle_state
    document = {"map_id": "draft", "evidence": [-1, 0, 100], "revision": 0}
    worker = SimpleNamespace(latest={"busy": False}, power_transition=False,
        home_mission=SimpleNamespace(active=False, home=SimpleNamespace(document=lambda: document), state=lambda: {"stage": "mapping"}),
        skill=None, continuous=None, ros_navigation=None, recorder=None, camera_frames={1: b"head-image"})

    async def call(operation):
        return operation(None)

    worker.call = call
    lab = SimpleNamespace(worker=worker, connections=0, lock=asyncio.Lock(), home_recording=None,
        state=lambda: {"run_id": "run", "stopped": True},
        agent=SimpleNamespace(active=False, recording_active=False, trace=lambda: {"events": []}))
    await preserve_idle_state(lab, tmp_path)
    snapshot = next(tmp_path.glob("*/snapshot.json"))
    assert json.loads(snapshot.read_text())["home_map"] == document
    assert snapshot.with_name("camera.png").read_bytes() == b"head-image"
    assert not list(tmp_path.rglob("*.sqlite3"))


async def test_real_unused_server_preserves_state_and_closes_physics_and_renderers(tmp_path, monkeypatch):
    import importlib
    import json
    import socket
    import httpx
    from websockets.asyncio.client import connect
    from backend.agent import FoundryConfig
    from backend.server import preserve_idle_state
    module = importlib.import_module("backend.app")
    monkeypatch.setenv("MILO_RENDERER", "enhanced")
    monkeypatch.setenv("MILO_HOME_MAP_STORE", str(tmp_path / "maps.sqlite3"))
    monkeypatch.setattr(module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(FoundryConfig, "from_environment", classmethod(lambda cls: cls()))
    activity = ServerActivity(module.app, idle_timeout_s=.5)
    server = IdleServer(uvicorn.Config(activity, log_level="warning", access_log=False), activity,
        lambda: lab_is_busy(module.lab), lambda: preserve_idle_state(module.lab, tmp_path / "archive"))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(90.):
            while not server.started:
                assert not serving.done()
                await asyncio.sleep(.01)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            initial = (await client.get("/api/state")).json()
            assert not initial["agent"]["active"] and initial["snapshot"]["simulated_time_s"] == 0
            worker = module.lab.worker
            resources = module.lab.render_resources
            renderer = resources.renderer
            processors = list(resources.processor._processes.values())
            async with connect(f"ws://127.0.0.1:{port}/api/live") as stream:
                await stream.recv()
                await asyncio.sleep(.8)
                assert not serving.done() and activity.connections
            response = await client.get("/api/state")
            assert response.status_code == 200
            await asyncio.sleep(.2)
            assert not serving.done()
        await asyncio.wait_for(serving, 30.)
        assert worker.closed and not worker.thread.is_alive()
        assert renderer.closed and renderer.process.poll() is not None
        assert all(not process.is_alive() for process in processors)
        snapshot = next((tmp_path / "archive").glob("*/snapshot.json"))
        saved = json.loads(snapshot.read_text())
        assert saved["state"]["run_id"] == initial["run_id"] and saved["state"]["stopped"]
        assert not saved["state"]["agent"]["active"]
        assert saved["state"]["snapshot"]["simulated_time_s"] == 0
        assert snapshot.with_name("camera.png").is_file()
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 30.)
        listener.close()