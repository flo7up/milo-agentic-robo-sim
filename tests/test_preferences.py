from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from backend.challenges import ChallengeLoad
from backend.preferences import PreferenceStore, PreferencesPatch


def test_preferences_survive_new_store_and_merge_without_scene_commands(tmp_path):
    path = tmp_path / "preferences.sqlite3"
    store = PreferenceStore(path)
    assert store.read() == {"version": 1, "preferences": {}, "scene": None}
    assert not path.exists()
    store.update(PreferencesPatch(turns=23, run_settings_open=True, goals={"standalone:bench": "Inspect the cube"}))
    other = PreferenceStore(path)
    other.update(PreferencesPatch(reasoning="medium", goals={"standalone:park": "Park safely"}))
    saved = store.read()
    assert saved["scene"] is None
    assert saved["preferences"] == {"turns": 23, "reasoning": "medium", "run_settings_open": True,
        "goals": {"standalone:bench": "Inspect the cube", "standalone:park": "Park safely"}}
    path.unlink()


def test_loaded_scene_is_separate_from_picker_draft(tmp_path):
    store = PreferenceStore(tmp_path / "preferences.sqlite3")
    store.save_scene(ChallengeLoad(challenge_id="bench"))
    store.update(PreferencesPatch(challenge_selection=ChallengeLoad(challenge_id="park", reuse_saved_map=False)))
    assert store.read()["scene"]["challenge_id"] == "bench"
    assert store.read()["preferences"]["challenge_selection"]["challenge_id"] == "park"


def test_parallel_preference_updates_retain_other_keys(tmp_path):
    path = tmp_path / "preferences.sqlite3"
    PreferenceStore(path).update(PreferencesPatch(turns=19))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda index: PreferenceStore(path).update(PreferencesPatch(goals={f"scenario:{index}": f"Goal {index}"})), range(12)))
    saved = PreferenceStore(path).read()["preferences"]
    assert saved["turns"] == 19 and len(saved["goals"]) == 12


@pytest.mark.parametrize("value", [{"turns": 0}, {"turns": 81}, {"interval": float("nan")}, {"handoff": "false"},
    {"navigation_mode": "invalid"}, {"api_key": "do-not-store"}, {"active": True}, {"run_id": "stale"},
    {"luna_endpoint": "https://user:password@example.com"}, {"luna_endpoint": "https://example.com?token=private"},
    {"luna_endpoint": "https://example.com#private"},
    {"goals": {"bench": "x" * 2001}}, {"head_pitch": 2}, {"loaded_scene": {"challenge_id": "park"}},
    {"position": [0., 0., 2.]}, {"joints": [2., 0., 0., 0., 0., 0.]}, {"joints": [float("nan")] * 6}])
def test_preferences_reject_invalid_values_credentials_and_motion_authority(value):
    with pytest.raises(ValidationError):
        PreferencesPatch.model_validate(value)