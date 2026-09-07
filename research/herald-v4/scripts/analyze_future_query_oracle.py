"""Fixed exposed-data mechanism gates for study047, never a fitted predictor."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from model_value_levels import concordance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(oracle, pending, losses):
    oracle = np.asarray(oracle, dtype=float)
    losses = np.asarray(losses, dtype=float)
    pending = np.asarray(pending, dtype=float)
    if oracle.shape != (64, 4) or losses.shape != (64, 4) or pending.shape != (64,):
        raise ValueError('Study047 requires exactly 64 prompts and four values each')
    if not all(np.isfinite(x).all() for x in (oracle, pending, losses)):
        raise ValueError('Nonfinite measurements or labels')
    mixed = np.ptp(losses, axis=1) != 0
    if mixed.sum() != 42:
        raise ValueError('Expected unchanged 42 mixed prompts from study045')
    prompt_risk = oracle.mean(axis=1)
    task_loss = losses.mean(axis=1)
    rho = float(spearmanr(prompt_risk, task_loss).statistic)
    control_rho = float(spearmanr(pending, task_loss).statistic)
    within = concordance(losses, oracle)
    i, j = np.triu_indices(4, 1)
    eligible = losses[:, i] != losses[:, j]
    ties = (oracle[:, i] == oracle[:, j]) & eligible
    def rank_ties(values):
        _, counts = np.unique(values, return_counts=True)
        return {'tied_groups': int((counts > 1).sum()), 'observations_in_ties': int(counts[counts > 1].sum())}
    return {
        'n': 64, 'mixed_prompts': 42,
        'within_prompt_concordance': within,
        'prompt_spearman': rho if np.isfinite(rho) else None,
        'pending_query_spearman': control_rho if np.isfinite(control_rho) else None,
        'eligible_value_pairs': int(eligible.sum()), 'tied_oracle_value_pairs': int(ties.sum()),
        'rank_ties': {k: rank_ties(v) for k, v in [('oracle', prompt_risk), ('pending', pending), ('loss', task_loss)]},
        'concordance_gate': bool(within is not None and within >= .81),
        'spearman_gate': bool(np.isfinite(rho) and rho >= .70),
        'passes': bool(within is not None and within >= .81 and np.isfinite(rho) and rho >= .70),
        'prompt_oracle': prompt_risk.tolist(), 'task_losses': task_loss.tolist(),
        'interpretation': 'Privileged exposed-data diagnostic; no B0 predictor or confirmation claim',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = json.loads((args.measurements / 'run.json').read_text())
    if run['status'] != 'completed' or run.get('failures'):
        raise ValueError('Collection did not complete cleanly')
    rows = []
    hashes = {}
    for item in run['prompts']:
        path = args.measurements / item['path']
        record = json.loads(path.read_text())
        if record['status'] != 'completed' or not record['checks'] or not all(record['checks'].values()):
            raise ValueError(f'Failed measurement {path}')
        hashes[str(path)] = sha(path)
        rows.append(record)
    priors = {}
    for path in args.prior.glob('*.json'):
        if path.name == 'run.json' or '.features.' in path.name:
            continue
        record = json.loads(path.read_text())
        rid = record['manifest_row']['id']
        if rid in priors:
            raise ValueError(f'Duplicate prior {rid}')
        priors[rid] = (path, record)
    ids = [r['manifest_row']['id'] for r in rows]
    expected = {f'value-level-v1-evaluation-{i:03d}' for i in range(64)}
    if len(ids) != 64 or set(ids) != expected or set(priors) != expected:
        raise ValueError('Exact study045 evaluation population required')
    labels = []
    for record in rows:
        path, prior = priors[record['manifest_row']['id']]
        if prior['status'] != 'completed' or not all(prior['checks'].values()):
            raise ValueError('Invalid prior evidence')
        got = [v['value_text'] for v in record['observation']['adapter']['values']]
        expected_values = [v['value_text'] for v in prior['observation']['adapter']['values']]
        if got != expected_values:
            raise ValueError('Value order mismatch')
        hashes[str(path)] = sha(path)
        labels.append(prior['per_value_signed_losses'])
    oracle = [r['observation']['per_value_oracle'] for r in rows]
    pending = [r['observation']['pending_query_error'] for r in rows]
    report = metrics(oracle, pending, labels)
    report.update(ids=ids, per_value_oracle=oracle, pending_query_error=pending,
                  per_value_signed_losses=labels, source_sha256=sha(__file__),
                  run_sha256=sha(args.measurements / 'run.json'), input_sha256=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('n', 'within_prompt_concordance', 'prompt_spearman', 'pending_query_spearman', 'passes')}))


if __name__ == '__main__':
    main()
