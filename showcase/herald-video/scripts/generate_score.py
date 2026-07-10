import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 48_000
DURATION_SECONDS = 85
OUTPUT = Path(__file__).resolve().parents[1] / "public/audio/score.wav"


def generate_score(path: Path = OUTPUT) -> None:
    total = SAMPLE_RATE * DURATION_SECONDS
    time = np.arange(total, dtype=np.float64) / SAMPLE_RATE
    score = np.zeros(total, dtype=np.float64)

    slow_motion = 0.68 + 0.32 * np.sin(2 * np.pi * time / 8.0)
    score += slow_motion * (
        0.11 * np.sin(2 * np.pi * 73.42 * time)
        + 0.055 * np.sin(2 * np.pi * 110.0 * time)
    )

    for start in (0.0, 18.0, 33.0, 52.0, 72.0, 81.0):
        local = time - start
        active = (local >= 0) & (local < 2.4)
        score[active] += (
            0.20
            * np.sin(2 * np.pi * 55.0 * local[active])
            * np.exp(-2.0 * local[active])
        )

    for start in np.arange(33.25, 52.0, 0.5):
        local = time - start
        active = (local >= 0) & (local < 0.055)
        score[active] += (
            0.055
            * np.sin(2 * np.pi * 1250.0 * local[active])
            * np.exp(-72.0 * local[active])
        )

    rollback = time - 40.5
    active = (rollback >= 0) & (rollback < 0.9)
    x = rollback[active]
    score[active] += 0.13 * np.sin(2 * np.pi * (480.0 * x - 155.0 * x * x)) * np.sin(
        np.pi * x / 0.9
    )

    commit = time - 46.5
    active = (commit >= 0) & (commit < 1.5)
    x = commit[active]
    envelope = np.sin(np.pi * x / 1.5) ** 2
    score[active] += envelope * (
        0.10 * np.sin(2 * np.pi * 261.63 * x)
        + 0.08 * np.sin(2 * np.pi * 392.0 * x)
    )

    fade_in = np.clip(time / 1.2, 0.0, 1.0)
    fade_out = np.clip((DURATION_SECONDS - time) / 2.5, 0.0, 1.0)
    score *= fade_in * fade_out
    score *= 0.72 / max(float(np.max(np.abs(score))), 1e-9)

    right = np.roll(score, int(0.011 * SAMPLE_RATE))
    right[: int(0.011 * SAMPLE_RATE)] = 0.0
    stereo = np.stack([score, right], axis=1)
    pcm = np.round(stereo * 32767.0).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm.tobytes())


if __name__ == "__main__":
    generate_score()
