import pytest

from generate_narration import validate_cues


def test_scene_clips_fit_their_frame_windows() -> None:
    cues = [
        {"id": "a", "from": 0, "duration": 180, "file": "a.mp3"},
        {"id": "b", "from": 210, "duration": 300, "file": "b.mp3"},
    ]
    validate_cues(cues, {"a": 5.5, "b": 9.0})


def test_rejects_clip_that_overruns_its_scene_window() -> None:
    cues = [{"id": "a", "from": 0, "duration": 180, "file": "a.mp3"}]
    with pytest.raises(ValueError, match="a exceeds its narration window"):
        validate_cues(cues, {"a": 5.9})
