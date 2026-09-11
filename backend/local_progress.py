import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


PROGRESS_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "navigation-progress"
TERMINAL_PHASES = {"completed", "failed", "interrupted"}


class NavigationProgressState(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    run_id: str
    checkpoint: str = Field(max_length=160)
    case: int = Field(ge=0, le=19)
    phase: Literal["checking", "loading", "warming", "running", "saving", "completed", "failed", "interrupted"] = "checking"
    started_at: float
    updated_at: float
    requests_completed: int = Field(default=0, ge=0, le=40)
    request_limit: int = Field(ge=1, le=40)
    success: bool | None = None
    outcome: str | None = Field(default=None, max_length=80)


class NavigationProgress:
    def __init__(self, checkpoint, case, request_limit):
        now = time.time()
        self.state = NavigationProgressState(run_id=str(uuid4()), checkpoint=Path(checkpoint).parent.name,
            case=case, request_limit=request_limit, started_at=now, updated_at=now)
        PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)
        self.path = PROGRESS_ROOT / f"{time.time_ns():020d}-{time.perf_counter_ns():020d}-{self.state.run_id}.json"
        self.update()

    def update(self, **changes):
        self.state = NavigationProgressState(**{**self.state.model_dump(), **changes, "updated_at": time.time()})
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(self.state.model_dump_json(), encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            return


@asynccontextmanager
async def track_navigation_progress(checkpoint, case, request_limit):
    progress = NavigationProgress(checkpoint, case, request_limit)

    async def heartbeat():
        while True:
            await asyncio.sleep(2)
            progress.update()

    task = asyncio.create_task(heartbeat())
    try:
        yield progress
    except BaseException:
        progress.update(phase="failed", success=False)
        raise
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def read_navigation_progress(now=None):
    paths = sorted(PROGRESS_ROOT.glob("*.json"), reverse=True)
    if not paths:
        return None
    path = paths[0]
    if path.stat().st_size > 16_384:
        raise ValueError("Invalid progress record")
    state = NavigationProgressState.model_validate_json(path.read_text(encoding="utf-8"))
    now = time.time() if now is None else now
    if state.phase not in TERMINAL_PHASES and not -5 <= now - state.updated_at <= 15:
        state.phase = "interrupted"
        state.success = False
    until = state.updated_at if state.phase in TERMINAL_PHASES else now
    return {**state.model_dump(), "elapsed_s": max(0, round(until - state.started_at))}