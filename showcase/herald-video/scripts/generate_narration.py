import json
import subprocess
from pathlib import Path
from typing import Any


FPS = 30
ROOT = Path(__file__).resolve().parents[1]
CUES_PATH = ROOT / "src/data/narration-cues.json"
VOICE = "en-US-AndrewMultilingualNeural"


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


def validate_cues(cues: list[dict[str, Any]], durations: dict[str, float]) -> None:
    for cue in cues:
        available = cue["duration"] / FPS - 0.25
        if durations[cue["id"]] > available:
            raise ValueError(f"{cue['id']} exceeds its narration window")


def generate_all() -> dict[str, float]:
    cues: list[dict[str, Any]] = json.loads(CUES_PATH.read_text())
    durations: dict[str, float] = {}
    for cue in cues:
        output = ROOT / "public" / cue["file"]
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "uvx",
                "--from",
                "edge-tts",
                "edge-tts",
                "--voice",
                VOICE,
                "--rate",
                cue.get("rate", "-5%"),
                "--text",
                cue["text"],
                "--write-media",
                str(output),
            ],
            check=True,
        )
        durations[cue["id"]] = probe_duration(output)
    validate_cues(cues, durations)
    return durations


if __name__ == "__main__":
    print(json.dumps(generate_all(), indent=2))
