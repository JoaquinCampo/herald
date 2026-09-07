# Recover HERALD after the Mac format

The authoritative research workspaces are preserved on Orion:

- `/clustergpu/home/jcampo/herald-v4`: current experiments, source, data and results.
- `/clustergpu/home/jcampo/herald-v3`: reference engine and archived v3 evidence.
- `/clustergpu/home/jcampo/herald-v2/.venv/bin/python`: existing Linux execution runtime.
- Pinned Qwen2.5-7B checkpoint: Hugging Face cache revision
  `a09a35458c702b33eeacc393d103063234e8bc28` under jcampo's remote home.

The GitHub repository is `https://github.com/joaquinCampo/herald`, branch
`feature/herald-v4-research`. This branch retains main's existing files and adds
`research/herald-v3` and `research/herald-v4`. Data, results, weights, virtualenvs
and credentials are intentionally absent from Git. They are not absent from the
Orion preservation copy.

## Restore a local checkout

```sh
git clone --branch feature/herald-v4-research https://github.com/joaquinCampo/herald.git
cd herald
```

Use `research/herald-v4` as the Codex project. Start by reading its `AGENTS.md`,
`FRAMEWORK.md`, `RESEARCH.md` and `STATE.md`. Research conclusions and active
work are in `experiments/`; the current objective is not yet scientifically met.
Do not reopen confirmation data or restart completed experiments blindly.

After restoring your SSH access to Orion, recover the local artifacts:

```sh
rsync -a orion:/clustergpu/home/jcampo/herald-v4/data/ research/herald-v4/data/
rsync -a orion:/clustergpu/home/jcampo/herald-v4/results/ research/herald-v4/results/
rsync -a orion:/clustergpu/home/jcampo/herald-v3/data/ research/herald-v3/data/
rsync -a orion:/clustergpu/home/jcampo/herald-v3/results/ research/herald-v3/results/
```

Do not copy a macOS virtualenv to Linux or a Linux virtualenv to macOS. Exact
installed distribution inventories are saved in `orion-runtime-packages.json`
and `mac-data-runtime-packages.json`. V3 also includes `pyproject.toml` and
`uv.lock`. Existing Linux experiments can continue using the unchanged remote
runtime and paths above. For local CPU analysis, create a fresh Python3.12 uv
virtualenv and install the relevant versions from the inventories; update local
absolute paths when a script requires them. GPU runs belong on Orion.

The local Codex heartbeat is not a server process and cannot keep running while
the Mac is erased/offline. Recreate research ownership in the restored Codex app,
using `RESEARCH.md` and `STATE.md`; keep the existing 5-minute milestone cadence
and preserve the user's GitHub/Orion synchronization authorization.

## Synchronization contract

Before ending a research milestone, copy all changed project files to Orion
using rsync without deletion, excluding only virtualenvs, Git internals and
rebuildable interpreter/test caches. Keep data/results on Orion. Copy code and
records into the matching `research/` folders of this branch, check the staged
files for credentials and accidental data/model additions, commit, and push.
Verify the GitHub branch SHA after pushing and run a checksum dry-run against
Orion. Never force-push, replace main, or delete remote historical artifacts.
If the remote branch has advanced, fetch and reconcile before writing.
