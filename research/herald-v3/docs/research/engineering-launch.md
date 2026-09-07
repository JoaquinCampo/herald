# Engineering acceptance launch

This runner executes the bounded eight prompt IFEval engineering slice. The
`--max-new-tokens` value is the total generated continuation budget, including
the shared 32 token boundary prefix. `--limit 1 --max-new-tokens 40` is the
first owner smoke; it exercises one prompt and is not the eight case result.

Build the portable prompt manifest on the Mac from the exposed development
inputs and the pinned local Arrow cache, then transfer the resulting file to
the task owned Orion directory:

```sh
cd /Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3
PYTHONPATH=src /Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v2/.venv/bin/python -c 'from herald_v3.engineering.prompts import load_prompt_manifest, write_prompt_manifest; m=load_prompt_manifest("/Users/joaquincamponario/orca/workspaces/herald-v2/cero/results/recovered/quality-risk-v1/audit-2026-09-04/hidden_state_development/inputs.json"); write_prompt_manifest(m, "data/engineering-prompts.json")'
rsync -av data/engineering-prompts.json orion:/clustergpu/home/jcampo/herald-v3/data/engineering-prompts.json
```

Run the one prompt smoke on Orion with the pinned snapshot and cached v2
runtime. The command records `run.json`, `run.log`, `prompt-manifest.json`,
and `artifacts.json` under the output directory:

```sh
cd /clustergpu/home/jcampo/herald-v3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /clustergpu/home/jcampo/herald-v2/.venv/bin/python scripts/run_engineering.py --model /clustergpu/home/jcampo/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --prompts data/engineering-prompts.json --output results/engineering-smoke-1 --max-new-tokens 40 --limit 1
```

After the smoke has been inspected by the owner, the eight prompt launch is:

```sh
cd /clustergpu/home/jcampo/herald-v3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /clustergpu/home/jcampo/herald-v2/.venv/bin/python scripts/run_engineering.py --model /clustergpu/home/jcampo/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --prompts data/engineering-prompts.json --output results/engineering-eight --max-new-tokens 1024 --limit 8
```

Each eligible row stores the complete reference and action token IDs and
decoded outputs, strict and loose instruction pass vectors, signed
`d_loose = q0_loose - qa_loose` and `d_strict = q0_strict - qa_strict`, and
the engine's boundary, cache, probe, timing, and gate evidence. Ineligible
early EOS and exceptions remain explicit rows. The convenience roster is
exposed development material, so this run supplies engineering evidence and
no inference claim.
