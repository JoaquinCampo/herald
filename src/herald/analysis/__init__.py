"""Post-training analyses for the hazard predictor.

Each module loads trained models + sweep results and writes
one JSON to `models/analysis/`. CLI wiring lives in `cli.py`;
`common.py` hosts the shared load-and-slice helpers reused
across modules.
"""
