import pytest


@pytest.fixture(autouse=True)
def isolate_workspace_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("MILO_PREFERENCES_STORE", str(tmp_path / "preferences.sqlite3"))
    monkeypatch.setenv("MILO_HOME_MAP_STORE", str(tmp_path / "homes.sqlite3"))