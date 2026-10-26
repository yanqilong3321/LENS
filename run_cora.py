"""Run the Cora portion of the LENS main table, without any API calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch_geometric

from lens import core
from lens.data import load_graph, load_pool

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dispatch(method, graph, pool):
    name, *items = method.split(':')
    kwargs = {key: float(value) for key, value in (v.split('=') for v in items)}
    return {'ce': core.m_ce, 'recipe': core.m_recipe, 'plug': core.m_plug}[name](graph, pool, **kwargs)


def metrics(graph, pool, prob):
    pred = prob.argmax(1)
    result = {}
    for key in ('val', 'test'):
        mask = pool[key]
        result[key] = 100 * float((pred[mask] == graph['gold'][mask]).float().mean())
        result[key + '_macro_f1'] = 100 * f1_score(
            graph['gold'][mask].cpu().numpy(), pred[mask].cpu().numpy(),
            average='macro', zero_division=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['GCN', 'SAGE', 'GAT', 'robust', 'all'], default='GCN')
    parser.add_argument('--seeds', default='360,361,362,363,364,365,366,367,368,369')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'cora')
    parser.add_argument('--smoke', action='store_true', help='Two epochs, seed 360, one candidate per row; not paper results')
    parser.add_argument('--methods', nargs='+', help='Optional exact method strings for debugging; not the full table')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    cfg = json.loads((ROOT / 'configs/cora.json').read_text())
    seeds = [360] if args.smoke else [int(s) for s in args.seeds.split(',')]
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= set(cfg['seeds']):
        parser.error('Seeds must be a nonempty unique subset of 360–369.')
    core.EPOCHS = 2 if args.smoke else cfg['epochs']
    suites = list(cfg['groups']) if args.suite == 'all' else [args.suite]
    graph, tensors, splits = load_graph(args.data_dir, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(dataset='cora', device=args.device, epochs=core.EPOCHS, threads=args.threads,
                    smoke=args.smoke, config_sha256=digest(ROOT / 'configs/cora.json'),
                    data_manifest_sha256=digest(args.data_dir / 'manifest.json'),
                    core_sha256=digest(ROOT / 'lens/core.py'),
                    runner_sha256=digest(__file__), loader_sha256=digest(ROOT / 'lens/data.py'),
                    python=platform.python_version(), torch=torch.__version__,
                    pyg=torch_geometric.__version__, numpy=np.__version__, api_calls=0)
    mf = args.output / 'run_manifest.json'
    if mf.exists() and json.loads(mf.read_text()) != manifest:
        raise ValueError('Output directory belongs to a different run configuration. Use a new --output.')
    mf.write_text(json.dumps(manifest, indent=2) + '\n')
    for suite in suites:
        core.BACKBONE = 'GCN' if suite == 'robust' else suite
        groups = cfg['groups'][suite]
        candidates = args.methods or sorted(set(
            m for values in groups.values() for m in (values[:1] if args.smoke else values)))
        allowed = set(m for values in groups.values() for m in values)
        if not set(candidates) <= allowed:
            raise ValueError(f'Unknown candidate for {suite}')
        path = args.output / f'{suite}.json'
        records = json.loads(path.read_text()) if path.exists() else {}
        for seed in seeds:
            pool = load_pool(graph, tensors, splits, seed)
            for method in candidates:
                key = f'{method}/{seed}'
                if key in records:
                    continue
                if str(args.device).startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                prob = dispatch(method, graph, pool)
                if str(args.device).startswith('cuda'):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                result = metrics(graph, pool, prob)
                result.update(seconds=elapsed, queries=len(pool['nodes']), api_calls=0)
                pred_dir = args.output / 'predictions' / suite
                pred_dir.mkdir(parents=True, exist_ok=True)
                token = hashlib.sha256(key.encode()).hexdigest()[:16]
                pred_path = pred_dir / f'{token}.npz'
                np.savez_compressed(pred_path, prediction=prob.argmax(1).cpu().numpy(),
                                    annotation_nodes=pool['nodes'].cpu().numpy(),
                                    validation_mask=pool['val'].cpu().numpy(), test_mask=pool['test'].cpu().numpy())
                result['prediction_file'] = str(pred_path.relative_to(args.output))
                records[key] = result
                tmp = path.with_suffix('.tmp')
                tmp.write_text(json.dumps(records, indent=2) + '\n')
                os.replace(tmp, path)
                print(f'{suite} seed={seed} {method}: val={result["val"]:.2f}, test={result["test"]:.2f} ({elapsed:.1f}s)', flush=True)
    print(f'Saved {args.output}. Use summarize.py for validation-selected table rows.')


if __name__ == '__main__':
    main()
