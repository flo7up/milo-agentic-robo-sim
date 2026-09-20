import json
import os
from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.challenges import ChallengeLoad


class PreferencesPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    interval: float | None = Field(default=None, ge=.25, le=30)
    turns: int | None = Field(default=None, ge=1, le=80)
    max_model_requests: int | None = Field(default=None, ge=1, le=200)
    max_model_tokens: int | None = Field(default=None, ge=1, le=2000000)
    kitchen_turns: int | None = Field(default=None, ge=1, le=80)
    kitchen_max_model_requests: int | None = Field(default=None, ge=1, le=200)
    kitchen_max_model_tokens: int | None = Field(default=None, ge=1, le=2000000)
    recording_enabled: bool | None = None
    recording_directory: str | None = Field(default=None, max_length=1024)
    reasoning: Literal["none", "low", "medium", "high"] | None = None
    luna_endpoint: str | None = Field(default=None, max_length=2048)
    luna_deployment: str | None = Field(default=None, max_length=128, pattern=r"^[\w.-]*$")
    mission_controller: Literal["luna", "qwen", "hybrid", "policy"] | None = None
    local_model_endpoint: str | None = Field(default=None, max_length=2048)
    local_model_tag: str | None = Field(default=None, max_length=120)
    control_mode: Literal["task", "exploration"] | None = None
    exploration_budget: int | None = Field(default=None, ge=1, le=300)
    handoff: bool | None = None
    skill_composer: bool | None = None
    navigation_backend: Literal["builtin", "nav2"] | None = None
    ai_routes: bool | None = None
    adaptive: bool | None = None
    mission_map_context: bool | None = None
    navigation_mode: Literal["luna_continuous", "luna_navigation"] | None = None
    inspector: Literal["conversation", "trace", "settings"] | None = None
    connection_open: bool | None = None
    run_settings_open: bool | None = None
    challenge_details_open: bool | None = None
    challenge_selection: ChallengeLoad | None = None
    goals: dict[Annotated[str, Field(min_length=1, max_length=160)], Annotated[str, Field(max_length=2000)]] | None = Field(default=None, max_length=80)
    compact_arms: bool | None = None
    graphics: Literal["standard", "enhanced"] | None = None
    axes: bool | None = None
    manual_open: bool | None = None
    manual_tab: Literal["drive", "head", "arms"] | None = None
    manual_arm: Literal["left", "right"] | None = None
    duration: float | None = Field(default=None, ge=.1, le=2)
    head_yaw: float | None = Field(default=None, ge=-1.5, le=1.5)
    head_pitch: float | None = Field(default=None, ge=-.7, le=1.15)
    opening: float | None = Field(default=None, ge=0, le=.11)
    force: float | None = Field(default=None, ge=1, le=35)
    position: list[Annotated[float, Field(ge=-1, le=1)]] | None = Field(default=None, min_length=3, max_length=3)
    joints: list[float] | None = Field(default=None, min_length=6, max_length=6)

    @field_validator("joints")
    @classmethod
    def joint_limits(cls, value):
        if value is not None and any(not -limit <= joint <= limit for joint, limit in zip(value, (1.8, 2.5, 2.7, 3, 2.5, 3))):
            raise ValueError("Joint targets must stay within the manual control limits")
        return value

    @field_validator("recording_directory")
    @classmethod
    def local_recording_directory(cls, value):
        if value is None:
            return value
        value = value.strip()
        if any(ord(character) < 32 for character in value) or value.startswith(("\\\\", "//")) or "://" in value:
            raise ValueError("Choose a local folder path, not a network path or URL")
        path = Path(value).expanduser()
        if path.drive and not path.is_absolute():
            raise ValueError("Use an absolute drive path or a workspace-relative folder")
        return value

    @field_validator("local_model_endpoint")
    @classmethod
    def local_endpoint(cls, value):
        if value is not None:
            from backend.agent import FoundryConfig
            return FoundryConfig.local_endpoint(value)
        return value

    @field_validator("luna_endpoint")
    @classmethod
    def credential_free_endpoint(cls, value):
        if value:
            parsed = urlsplit(value)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Save only an endpoint without credentials, query parameters or fragments")
        return value


class PreferenceStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("MILO_PREFERENCES_STORE") or
            Path(__file__).resolve().parents[1] / ".runtime/preferences.sqlite3")

    def read(self):
        if not self.path.exists():
            return {"version": 1, "preferences": {}, "scene": None}
        with closing(sqlite3.connect(f"file:{self.path.resolve().as_posix()}?mode=ro", uri=True, timeout=5)) as connection:
            values = {name: json.loads(value) for name, value in connection.execute("SELECT name, value FROM preferences")}
        scene = values.pop("loaded_scene", None)
        preferences = PreferencesPatch.model_validate(values).model_dump(mode="json", exclude_none=True)
        return {"version": 1, "preferences": preferences,
            "scene": ChallengeLoad.model_validate(scene).model_dump() if scene is not None else None}

    def update(self, patch: PreferencesPatch):
        self._write(patch.model_dump(mode="json", exclude_unset=True, exclude_none=True))
        return self.read()

    def save_scene(self, selection: ChallengeLoad):
        self._write({"loaded_scene": selection.model_dump(mode="json")})

    def recording_directories(self):
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(f"file:{self.path.resolve().as_posix()}?mode=ro", uri=True, timeout=5)) as connection:
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='recording_directories'").fetchone()
            return [row[0] for row in connection.execute("SELECT path FROM recording_directories ORDER BY rowid DESC LIMIT 100")] if exists else []

    def _write(self, changes):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS preferences (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("BEGIN IMMEDIATE")
            if changes.get("recording_directory"):
                directory = Path(changes["recording_directory"]).expanduser()
                if not directory.is_absolute():
                    directory = Path(__file__).resolve().parents[1] / directory
                directory = directory.resolve()
                changes["recording_directory"] = str(directory)
                connection.execute("CREATE TABLE IF NOT EXISTS recording_directories (path TEXT PRIMARY KEY)")
                connection.execute("INSERT OR IGNORE INTO recording_directories(path) VALUES (?)", (str(directory),))
            if "goals" in changes:
                previous = connection.execute("SELECT value FROM preferences WHERE name = 'goals'").fetchone()
                changes["goals"] = {**(json.loads(previous[0]) if previous else {}), **changes["goals"]}
                PreferencesPatch(goals=changes["goals"])
            connection.executemany("INSERT INTO preferences(name, value) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                [(name, json.dumps(value, allow_nan=False)) for name, value in changes.items()])