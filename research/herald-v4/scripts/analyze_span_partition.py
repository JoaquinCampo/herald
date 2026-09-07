"""Summarize the fixed causal partition, never fit a predictor."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {f'ruler-ea-dev-v1-niah_single_2-{i:03}' for i in (0, 1, 2, 3, 4, 5, 8, 10, 11)}
rows = []
for part in ('first', 'rest'):
    folder = ROOT / f'results/span-partition-v1-{part}'
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'completed' and not run['failures']
    for item in run['prompts']:
        record = json.loads((folder / item['path']).read_text())
        assert record['status'] == 'completed' and record['checks']['all_checks_pass']
        scores = {name: b['score']['score_fraction'] for name, b in record['branches'].items()}
        swaps = {name: sum(h['k'] for layer in record['branches'][name]['swap_audit'] for h in layer) for name in ('value_rescue', 'context_rescue', 'value_control', 'context_control')}
        rows.append({'id': item['id'], 'scores': scores, 'swaps': swaps})
assert len(rows) == len({r['id'] for r in rows}) == 9
assert {r['id'] for r in rows} == EXPECTED
failed = [r for r in rows if not r['id'].endswith('-001')]
guard = next(r for r in rows if r['id'].endswith('-001'))
assert all(r['scores']['reference_shared_boundary'] == r['scores']['oracle_span_rescue'] == 1 for r in rows)
assert all(r['scores']['standard_knorm_0.10'] == 0 for r in failed)
assert guard['scores']['standard_knorm_0.10'] == 1
counts = {name: sum(r['scores'][name] == 1 for r in failed) for name in failed[0]['scores']}
patterns = Counter(('both' if r['scores']['value_rescue'] and r['scores']['context_rescue'] else 'value_only' if r['scores']['value_rescue'] else 'context_only' if r['scores']['context_rescue'] else 'neither') for r in failed)
gates = {p: counts[p + '_rescue'] >= 6 and counts[p + '_control'] == 0 and guard['scores'][p + '_rescue'] == guard['scores'][p + '_control'] == 1 for p in ('value', 'context')}
report = {'scope': 'Eight exposed B0 .10 failures plus healthy001; oracle causal study, no predictor', 'rescued_of_eight': counts, 'patterns': dict(patterns), 'healthy_guard': guard['scores'], 'recurrent_mechanism_gates': gates, 'rows': sorted(rows, key=lambda r: r['id'])}
(ROOT / 'results/span-partition-summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
