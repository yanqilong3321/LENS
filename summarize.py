"""Rebuild the Cora main table using validation-only per-seed selection."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def select(records, candidates, seed):
    missing = [m for m in candidates if f'{m}/{seed}' not in records]
    if missing:
        raise ValueError(f'Incomplete candidates for seed {seed}: {len(missing)} missing')
    # max preserves candidate order on ties, as did the original table builder.
    method = max(candidates, key=lambda m: records[f'{m}/{seed}']['val'])
    return method, records[f'{method}/{seed}']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'outputs/cora')
    p.add_argument('--seeds', default='360,361,362,363,364,365,366,367,368,369')
    p.add_argument('--suite', choices=['GCN','SAGE','GAT','robust','all'], default='all')
    p.add_argument('--output', type=Path, help='Optional JSON report')
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    cfg = json.loads((ROOT / 'configs/cora.json').read_text())
    suites = list(cfg['groups']) if a.suite == 'all' else [a.suite]
    rows = {}
    print('| Learner | Variant | Accuracy (%) | Macro-F1 (%) |')
    print('|---|---|---:|---:|')
    for suite in suites:
        records = json.loads((a.input / f'{suite}.json').read_text())
        for label, candidates in cfg['groups'][suite].items():
            selected = [select(records, candidates, seed) for seed in seeds]
            vals = np.array([r['test'] for _, r in selected])
            mean, std = float(vals.mean()), float(vals.std(ddof=1)) if len(vals)>1 else None
            acc = f'{mean:.2f} ± {std:.2f}' if std is not None else f'{mean:.2f}'
            f1s = [r.get('test_macro_f1') for _, r in selected]
            f1 = '—'
            if all(x is not None for x in f1s):
                f1 = f'{np.mean(f1s):.2f}' + (f' ± {np.std(f1s,ddof=1):.2f}' if len(f1s)>1 else '')
            rows[f'{suite}/{label}'] = dict(mean=mean, std=std, per_seed={str(s):dict(method=m, **r) for s,(m,r) in zip(seeds,selected)})
            print(f'| {suite} | {label} | {acc} | {f1} |')
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
