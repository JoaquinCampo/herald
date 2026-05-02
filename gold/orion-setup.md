# Orion: sweep-host setup notes

Operational reference for the GPU sweep host. Update when the
environment changes; do not duplicate Phase-level decisions from the
research plan here.

## Connection

- SSH alias: `ssh orion`
- Project root: `/clustergpu/home/jcampo/herald`
- Layout: flat rsync target (NOT a git checkout). Sync from local Mac
  with `rsync -av <subdir>/ orion:.../<subdir>/` per top-level path.
  Never `--delete`. Never sync `.venv/`, `results/`, `models/`, logs.

## Hardware

- NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CUDA 13.1
- Always check `nvidia-smi` for stale processes before launching a
  job; a crashed CUDA process can leave the GPU in a dirty state that
  poisons subsequent runs.

## Python environment

- Python 3.12 in `.venv/`
- Herald is wired into the venv via a `.pth` file (no
  `pip install -e .`); editing `src/herald/...` on disk takes effect
  on the next process.
- `pytest` and `poethepoet` are intentionally NOT installed on Orion.
  CPU checks (`poe check`, `pytest`) run on the Mac. Orion runs
  standalone smoke / profiling scripts under `scripts/`.

## Network and proxy

Orion has no direct outbound internet. PyPI / Hugging Face traffic
goes through an SSH reverse tunnel from the Mac on port 18080.

1. On the **Mac**, bring the proxy up:

   ```sh
   # Local CONNECT proxy (any tiny CONNECT-capable HTTPS forwarder).
   python /tmp/proxy.py &
   ssh -R 18080:127.0.0.1:18080 -N -f orion
   ```

2. On **Orion**, before running anything that touches the hub:

   ```sh
   export HTTPS_PROXY=http://127.0.0.1:18080
   export HTTP_PROXY=http://127.0.0.1:18080
   ```

   (Lower-case forms work too. The env is per-shell; persistent
   placement in `~/.bashrc` is fine if you accept that every Orion
   shell will try the proxy.)

3. To verify:

   ```sh
   curl -sS --max-time 5 https://huggingface.co/api/whoami | head
   ```

   Returns `{"error":"Invalid username or password."}` if the tunnel
   is up. Tear the tunnel down with `pkill -f "ssh -R 18080"` when
   done.

## Dataset prefetch

All benchmark data is fetched explicitly via the prefetcher; no
implicit downloads inside Phase 1 execution.

```sh
.venv/bin/python scripts/prefetch_datasets.py --list      # audit
.venv/bin/python scripts/prefetch_datasets.py --dry-run   # confirm
.venv/bin/python scripts/prefetch_datasets.py             # execute
```

`scripts/prefetch_datasets.py` reads `herald.tasks.PHASE1_DATASET_SPECS`
as the single source of truth (name / config / split / revision).
Add a benchmark by extending that constant; do not duplicate
coordinates here.

## Sanity checks before any sweep

1. `nvidia-smi` shows no other processes.
2. `ls ~/.cache/huggingface/datasets/` covers the targets you need.
3. Latest local code is rsync'd:
   `rsync -av src/ orion:/clustergpu/home/jcampo/herald/src/`
4. Equivalence smoke (regenerates one Phase 0 baseline run with the
   current code path):
   `.venv/bin/python scripts/phase1_equivalence_check.py`
