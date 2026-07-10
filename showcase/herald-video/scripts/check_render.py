import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "out/herald-openai-showcase.mp4"


def validate_probe(probe: dict[str, Any]) -> None:
    duration = float(probe["format"]["duration"])
    if not 84.95 <= duration <= 85.05:
        raise ValueError(f"duration must be 85 seconds, got {duration:.6f}")

    video = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("missing video stream")
    if video.get("codec_name") != "h264":
        raise ValueError("video must use h264")
    if (video.get("width"), video.get("height")) != (1920, 1080):
        raise ValueError("video must be 1920x1080")
    if video.get("r_frame_rate") != "30/1":
        raise ValueError("video must be 30 fps")

    audio = next((stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None)
    if audio is None:
        raise ValueError("missing audio stream")
    if audio.get("codec_name") != "aac":
        raise ValueError("audio must use aac")
    if audio.get("sample_rate") != "48000":
        raise ValueError("audio must use 48 kHz")


def main() -> None:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_streams",
            "-of",
            "json",
            str(MASTER),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    validate_probe(json.loads(completed.stdout))
    print("PASS: final media contract")


if __name__ == "__main__":
    main()
