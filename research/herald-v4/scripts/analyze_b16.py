"""Evaluate only the locked delayed B16 estimand after its viability gate."""
import json
from pathlib import Path

import numpy as np

from analyze_retrieval_probe import (
    auc, baseline_loo, bootstrap_gain, exact_auc_permutation, fit_loo,
    metrics, paired_wins,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for part in ('first', 'rest'):
        directory = ROOT / f'results/b16-v1-{part}'
        run = json.loads((directory / 'run.json').read_text())
        if run.get('failures'):
            raise ValueError('Recorded B16 failure blocks interpretation')
        for item in run['prompts']:
            record = json.loads((directory / item['path']).read_text())
            if record['status'] != 'completed':
                rows.append({'prompt_id': item['id'], 'eligible': False,
                             'reason': record.get('reason', record['status'])})
                continue
            if not record['checks']['all_checks_pass']:
                raise ValueError('B16 controls failed')
            rows.append({'prompt_id': item['id'], 'eligible': True,
                         'signed_loss': record['signed_loss'],
                         'z': record['feature']['z']})
    assert len(rows) == len({r['prompt_id'] for r in rows}) == 12
    complete = all(r['eligible'] for r in rows)
    positive = sum(r.get('signed_loss', 0) > 0 for r in rows)
    zero = sum(r.get('signed_loss') == 0 for r in rows)
    viable = complete and positive >= 3 and zero >= 3
    report = {'scope': 'Exposed 12-prompt NIAH .10 B16 development; changed estimand.',
              'all_boundaries_valid': complete, 'positive_labels': positive,
              'zero_labels': zero, 'label_viable': viable, 'proceed': False}
    if viable:
        y = np.array([r['signed_loss'] for r in rows])
        baseline = baseline_loo(rows)
        prediction, fits = fit_loo(rows, 'z')
        gain = 1 - np.mean((prediction-y)**2) / np.mean((baseline-y)**2)
        wins = paired_wins(y, baseline, prediction)
        raw_auc = auc(np.array([r['z'] for r in rows]), y > 0)
        report.update({'baseline': metrics(y, baseline),
                       'feature': metrics(y, prediction), 'gain_fraction': float(gain),
                       'prompt_wins': wins, 'raw_positive_auc': raw_auc,
                       'permutation': exact_auc_permutation([r['z'] for r in rows], y > 0),
                       'bootstrap_fixed_oof_pairs': bootstrap_gain(y, baseline, prediction),
                       'gates': {'mse_gain_10_percent': bool(gain >= .10),
                                 'eight_prompt_wins': wins['model_wins'] >= 8,
                                 'auc_0_80': raw_auc >= .80}, 'loo_fits': fits})
        report['proceed'] = all(report['gates'].values())
        for i, row in enumerate(rows):
            row.update(baseline_prediction=float(baseline[i]), prediction=float(prediction[i]))
    output = ROOT / 'results/b16-model'
    output.mkdir(exist_ok=True)
    (output / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'loo_fits'}, indent=2))


if __name__ == '__main__':
    main()
