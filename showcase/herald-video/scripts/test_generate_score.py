import wave
from pathlib import Path

from generate_score import DURATION_SECONDS, OUTPUT, SAMPLE_RATE, generate_score


def test_score_is_85_second_stereo_without_clipping(tmp_path: Path) -> None:
    path = tmp_path / "score.wav"
    generate_score(path)
    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 2
        assert audio.getframerate() == SAMPLE_RATE
        assert audio.getnframes() == SAMPLE_RATE * DURATION_SECONDS
    assert path.stat().st_size > 1_000_000
    assert OUTPUT.name == "score.wav"
