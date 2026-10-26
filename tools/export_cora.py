"""Export public Cora tensors and numeric LLM predictions from trusted local caches.

This is a maintainer utility; the release already includes its output. Never pass
untrusted pickle/.pt inputs. Raw prompts, completions and credentials are not exported.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.utils import to_undirected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('data'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    graph = torch.load(args.graph, map_location='cpu', weights_only=False)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    x = F.normalize(graph.x.float(), dim=1).numpy()
    n = len(x)
    edges = to_undirected(graph.edge_index, num_nodes=n).numpy()
    pred = cache['pred'].long().numpy()
    conf = cache['conf'].float().numpy()
    gold = graph.y.view(-1).long().numpy()
    assert n == 2708 and x.shape[1] == 384
    assert pred.shape == gold.shape == conf.shape == (n,)
    np.savez_compressed(args.out / 'cora.npz', x=x, edge_index=edges,
                        gold=gold, llm_labels=pred, llm_confidence=conf)
    pools = {}
    for seed in range(360, 370):
        rng = np.random.default_rng(20261017 + seed)
        val = np.zeros(n, dtype=bool)
        val[rng.choice(n, 500, replace=False)] = True
        cand = np.where((~val) & (pred >= 0))[0]
        km = KMeans(350, n_init=1, random_state=seed, max_iter=50).fit(x[cand])
        distance = ((x[cand] - km.cluster_centers_[km.labels_]) ** 2).sum(1)
        chosen = []
        for c in range(350):
            idx = np.where(km.labels_ == c)[0]
            if idx.size:
                chosen.append(int(cand[idx[np.argmin(distance[idx])]]))
        nodes = sorted(set(chosen))
        assert len(nodes) == 350
        pools[str(seed)] = {'nodes': nodes, 'validation': np.where(val)[0].tolist()}
    (args.out / 'splits.json').write_text(json.dumps(pools, indent=2) + '\n')
    manifest = {'dataset': 'Cora', 'nodes': n, 'features': int(x.shape[1]),
                'classes': int(gold.max()) + 1, 'budget': 350,
                'validation_size': 500, 'seeds': list(range(360, 370)),
                'source_graph_sha256': hashlib.sha256(args.graph.read_bytes()).hexdigest(),
                'source_cache_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                'files': {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
                          for name in ['cora.npz', 'splits.json']}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
