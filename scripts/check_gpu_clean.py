"""Refuse to start a sweep if the GPU is dirty.

Treats > 200 MB of VRAM held by other processes as 'dirty'. Exits 1
on dirty, 0 on clean. The orchestrator calls this as a precheck.
"""

import subprocess
import sys


def main() -> int:
    out = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        .decode()
        .strip()
    )
    if not out:
        return 0
    dirty = []
    for line in out.splitlines():
        pid, mem = (s.strip() for s in line.split(","))
        if int(mem) > 200:
            dirty.append((pid, mem))
    if dirty:
        print(
            "GPU dirty; existing processes:\n"
            + "\n".join(f"  pid={p} used={m} MiB" for p, m in dirty),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
