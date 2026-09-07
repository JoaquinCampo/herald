"""Study051: exposed-data cross-action diagnostic, with no predictor fit."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from score_ruler_pilot import score_prediction

ROOT = Path(__file__).resolve().parents[1]
RATES = [.05, .1, .2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(matrix):
    d = np.asarray(matrix, dtype=float)
    n = len(d)
    assert d.shape == (n, 3) and n > 1 and np.isfinite(d).all()
    baseline = (d.sum(axis=0, keepdims=True) - d) / (n - 1)
    centered = d - baseline
    predicted = (centered.sum(axis=1, keepdims=True) - centered) / 2
    base_mse = float(np.mean(centered**2))
    oracle_mse = float(np.mean((centered - predicted)**2))
    gain = 1 - oracle_mse / base_mse if base_mse > 0 else None
    tau = [float(kendalltau(d[:, i], d[:, i + 1], variant='b').statistic) for i in range(2)]
    return {'n': n, 'matrix': d.tolist(), 'monotone_prompts': int(np.sum(np.all(np.diff(d, axis=1) >= 0, axis=1))),
            'adjacent_tau_b': [v if np.isfinite(v) else None for v in tau],
            'baseline_mse': base_mse, 'privileged_mse': oracle_mse, 'mse_gain': gain,
            'loo_task_rate_means': baseline.tolist(), 'centered_target': centered.tolist(),
            'privileged_centered_prediction': predicted.tolist()}


def main():
    output = ROOT / 'results/action-identifiability-summary.json'
    assert not output.exists()
    groups = {}; hashes = {}; ids = set()
    for dirname in ('ea-dev-v1-first', 'ea-dev-v1-rest'):
        directory = ROOT / 'results' / dirname
        run = json.loads((directory / 'run.json').read_text())
        assert run['status'] == 'completed' and not run['failures']
        hashes[str(directory / 'run.json')] = sha(directory / 'run.json')
        for item in run['prompts']:
            path = directory / item['path']; r = json.loads(path.read_text())
            assert r['status'] == 'completed' and all(r['checks'].values())
            rid = r['manifest_row']['id']; assert rid not in ids; ids.add(rid)
            answers = r['manifest_row']['answers']
            ref = score_prediction(r['reference']['text'], answers)['score_fraction']
            byrate = {float(a['action']['removal_fraction']): a for a in r['arms'].values()}
            assert set(byrate) == {0., *RATES}
            assert byrate[0.]['continuation']['token_ids'] == r['reference']['token_ids']
            losses = [ref - score_prediction(byrate[rate]['continuation']['text'], answers)['score_fraction'] for rate in RATES]
            task = r['manifest_row']['task']
            groups.setdefault(task, []).append({'id': rid, 'reference': ref, 'losses': losses})
            hashes[str(path)] = sha(path)
    assert len(ids) == 20
    results = {task: {**diagnose([r['losses'] for r in rows]), 'rows': rows} for task, rows in groups.items()}
    primary = results['niah_single_2']; assert primary['n'] == 12
    assert len(results) == 2 and sorted(r['n'] for r in results.values()) == [8, 12]
    gates = {'monotonicity': primary['monotone_prompts'] >= 8,
             'adjacent_tau': all(v is not None and v >= .4 for v in primary['adjacent_tau_b']),
             'privileged_mse_gain': primary['mse_gain'] is not None and primary['mse_gain'] >= .2}
    report = {'rates': RATES, 'tasks': results, 'gates': gates, 'passes': all(gates.values()),
              'source_sha256': sha(Path(__file__)), 'input_sha256': hashes,
              'scope': 'Exposed privileged assumption diagnostic; no decision-time predictor or confirmation claim'}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'gates': gates, 'passes': report['passes'], 'tasks': {t: {k: r[k] for k in ('n', 'monotone_prompts', 'adjacent_tau_b', 'mse_gain')} for t, r in results.items()}}))


if __name__ == '__main__':
    main()
