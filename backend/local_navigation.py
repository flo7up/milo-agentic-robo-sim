import asyncio
import base64
from contextlib import suppress
import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from backend.contracts import StrictModel
from backend.policy import PolicyTicket
from backend.smolvla_server import PolicyRequest


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / ".runtime/navigation-recovery-6500/checkpoint"
MODEL_PYTHON = ROOT / ".runtime/smolvla-env/Scripts/python.exe"
TASK = "Drive into the green floor bay and stop. Keep clear of the posts and walls."


class NavigationRequest(PolicyRequest):
    instruction: str = Field(min_length=1, max_length=2000)


class NavigationReset(StrictModel):
    reset: Literal[True]


class LocalNavigationClient:
    def __init__(self, checkpoint=CHECKPOINT):
        self.process = None
        self.log = None
        self.instruction = TASK
        self.checkpoint = Path(checkpoint)

    @property
    def alive(self):
        return self.process is not None and self.process.returncode is None

    @staticmethod
    def check_available():
        if not MODEL_PYTHON.is_file() or not (CHECKPOINT / "model.safetensors").is_file():
            raise ValueError("Local navigation needs the Windows model environment and navigation-recovery-6500 checkpoint.")

    async def start(self):
        self.check_available()
        directory = ROOT / ".runtime/local-navigation"
        directory.mkdir(parents=True, exist_ok=True)
        self.log = (directory / f"{uuid4()}.log").open("w", encoding="utf-8")
        spawning = asyncio.create_task(asyncio.create_subprocess_exec(str(MODEL_PYTHON), "-u", "-m", "scripts.navigation_policy",
            "--stage", "inference", "--checkpoint", str(self.checkpoint), "--seed", "716", cwd=ROOT,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.log, limit=2_500_000))
        try:
            self.process = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            self.process = await spawning
            raise
        try:
            ready = await self.reply(300)
        except TimeoutError as error:
            raise ValueError("Local model startup exceeded five minutes; robot stayed stationary. Check .runtime/local-navigation for the loading stage.") from error
        if ready.get("ready") is not True:
            raise ValueError("Local navigation model did not become ready.")

    async def reply(self, timeout):
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
        if not line:
            raise ValueError("Local navigation model exited. Check .runtime/local-navigation for its startup log.")
        return json.loads(line)

    async def predict(self, observation, image):
        from scripts.navigation_policy import bounded_velocity
        ticket = PolicyTicket(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
            revision=observation.navigation.revision if observation.navigation else 0, observation_seq=observation.seq)
        request = NavigationRequest(ticket=ticket, observation=observation, image=base64.b64encode(image).decode("ascii"), instruction=self.instruction)
        self.process.stdin.write((request.model_dump_json() + "\n").encode())
        await self.process.stdin.drain()
        reply = await self.reply(10)
        if reply.get("ticket") != ticket.model_dump():
            raise ValueError("Local navigation returned a stale or mismatched observation ticket.")
        action, saturated = bounded_velocity(reply["action"])
        return {"action": action, "raw_action": reply["action"], "saturated_axes": saturated}

    async def reset(self):
        self.process.stdin.write((NavigationReset(reset=True).model_dump_json() + "\n").encode())
        await self.process.stdin.drain()
        if await self.reply(10) != {"reset": True}:
            raise ValueError("Local navigation session reset was not acknowledged.")

    async def close(self):
        try:
            stdin = getattr(self.process, "stdin", None)
            if stdin:
                stdin.close()
            if self.process and self.process.returncode is None:
                if os.name == "nt":
                    killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(self.process.pid), "/T", "/F",
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                    await killer.wait()
                else:
                    with suppress(ProcessLookupError):
                        self.process.kill()
                await asyncio.wait_for(self.process.wait(), 10)
            if stdin:
                with suppress(BrokenPipeError, ConnectionResetError):
                    await asyncio.wait_for(stdin.wait_closed(), 5)
            stdout = getattr(self.process, "stdout", None)
            if stdout:
                await asyncio.wait_for(stdout.read(), 5)
        finally:
            if self.log:
                self.log.close()


class ResidentNavigationModel:
    def __init__(self, client_factory=LocalNavigationClient):
        self.client_factory = client_factory
        self.client = None
        self.lock = asyncio.Lock()
        self.tasks = set()
        self.phase = "unloaded"
        self.load_count = 0
        self.closing = False

    def check_available(self):
        self.client_factory.check_available()

    def __call__(self):
        return ResidentNavigationSession(self)

    def public(self):
        phase = self.phase
        if self.client and phase in {"ready", "inferencing"} and not self.client.alive:
            phase = "error"
        process = getattr(self.client, "process", None)
        return {"phase": phase, "checkpoint": CHECKPOINT.parent.name, "load_count": self.load_count,
                "process_id": process.pid if process and self.client.alive else None}

    def _finished(self, task):
        self.tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def submit(self, session, observation=None, image=None):
        if session.closed or self.closing:
            raise asyncio.CancelledError
        task = asyncio.create_task(self._operate(session, observation, image))
        self.tasks.add(task)
        task.add_done_callback(self._finished)
        return await asyncio.shield(task)

    async def _operate(self, session, observation, image):
        async with self.lock:
            if session.closed or self.closing:
                raise asyncio.CancelledError
            try:
                if self.client and not self.client.alive:
                    await self.client.close()
                    self.client = None
                if self.client is None:
                    self.client = self.client_factory()
                    self.phase = "loading"
                    self.load_count += 1
                    await self.client.start()
                    self.phase = "ready"
                if session.closed:
                    raise asyncio.CancelledError
                if observation is None:
                    await self.client.reset()
                    return
                self.phase = "inferencing"
                self.client.instruction = session.instruction
                async with asyncio.timeout(10):
                    result = await self.client.predict(observation, image)
                if session.closed:
                    raise asyncio.CancelledError
                return result
            except Exception:
                self.phase = "error"
                if self.client:
                    await self.client.close()
                    self.client = None
                raise
            finally:
                if self.phase == "inferencing":
                    self.phase = "ready"

    async def close(self):
        self.closing = True
        try:
            pending = list(self.tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if self.client:
                await self.client.close()
        finally:
            self.client = None
            self.phase = "unloaded"
            self.closing = False


class ResidentNavigationSession:
    def __init__(self, resident):
        self.resident = resident
        self.instruction = TASK
        self.closed = False

    async def start(self):
        await self.resident.submit(self)

    async def predict(self, observation, image):
        return await self.resident.submit(self, observation, image)

    async def close(self):
        self.closed = True