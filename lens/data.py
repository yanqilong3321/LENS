"""Portable data loading; no pickle loading or external API calls at runtime."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def load_graph(data_dir, device='cpu'):
    root = Path(data_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Data checksum mismatch: {name}')
    with np.load(root / 'cora.npz', allow_pickle=False) as a:
        tensors = {k: torch.from_numpy(a[k].copy()).to(device) for k in a.files}
    graph = dict(ds='cora', n=len(tensors['x']), x=tensors['x'],
                 e=tensors['edge_index'], gold=tensors['gold'], C=7, device=device)
    return graph, tensors, json.loads((root / 'splits.json').read_text())


def load_pool(graph, tensors, splits, seed):
    if str(seed) not in splits:
        raise ValueError('This release contains the paper seeds 360–369 only.')
    dev, n = graph['device'], graph['n']
    split = splits[str(seed)]
    nodes = torch.tensor(split['nodes'], dtype=torch.long, device=dev)
    val = torch.zeros(n, dtype=torch.bool, device=dev)
    val[split['validation']] = True
    ann = torch.zeros_like(val); ann[nodes] = True
    if len(nodes) != 350 or val.sum() != 500 or (ann & val).any():
        raise ValueError('Invalid annotation/validation split')
    y = torch.full((n,), -1, dtype=torch.long, device=dev)
    y[nodes] = tensors['llm_labels'][nodes]
    conf = torch.zeros(n, device=dev)
    conf[nodes] = tensors['llm_confidence'][nodes]
    src = torch.zeros(n, dtype=torch.long, device=dev); src[nodes] = 1
    return dict(seed=seed, nodes=nodes, y=y, conf=(conf / 100).clamp(0, 1),
                val=val, test=~ann & ~val, src=src,
                gnn_q=torch.full((n,), -1, dtype=torch.long, device=dev))
