"""LENS training kernels extracted from the paper main-experiment implementation.

The computation is preserved; historical search drivers and LoCLE imports are removed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

EPOCHS = 200
BACKBONE = "GCN"

class GCN(torch.nn.Module):
    def __init__(self, d, c, h=64, p=0.5):
        super().__init__()
        self.c1, self.c2, self.p = GCNConv(d, h), GCNConv(h, c), p

    def forward(self, x, e):
        return self.c2(F.dropout(F.relu(self.c1(F.dropout(x, 0.0, self.training), e)), self.p, self.training), e)


class GAT(torch.nn.Module):
    def __init__(self, d, c, h=8, heads=8, p=0.6):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.c1, self.c2, self.p = GATConv(d, h, heads=heads, dropout=p), GATConv(h * heads, c, heads=1, dropout=p), p

    def forward(self, x, e):
        x = F.dropout(x, self.p, self.training)
        return self.c2(F.dropout(F.elu(self.c1(x, e)), self.p, self.training), e)


class SAGE(torch.nn.Module):
    def __init__(self, d, c, h=64, p=0.5):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.c1, self.c2, self.p = SAGEConv(d, h), SAGEConv(h, c), p

    def forward(self, x, e):
        return self.c2(F.dropout(F.relu(self.c1(x, e)), self.p, self.training), e)


def new_model(G, seed):
    torch.manual_seed(seed)
    cls = {"GCN": GCN, "GAT": GAT, "SAGE": SAGE}[BACKBONE]
    return cls(G["x"].shape[1], G["C"]).to(G["device"])


def train(G, nodes, loss_fn, seed, epochs=None, model=None, hook=None):
    epochs = EPOCHS if epochs is None else epochs
    model = model or new_model(G, seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(G["x"], G["e"])
        loss = loss_fn(logits[nodes], ep)
        loss.backward(); opt.step()
        if hook:
            hook(model, ep)
    model.eval()
    with torch.no_grad():
        return model, F.softmax(model(G["x"], G["e"]), 1)


def ce_targets(logits, targets, w=None):
    l = -(targets * F.log_softmax(logits, 1)).sum(1)
    return (l * w).sum() / w.sum() if w is not None else l.mean()


def onehot(y, C):
    return F.one_hot(y, C).float()


def m_ce(G, P):
    t = onehot(P["y"][P["nodes"]], G["C"])
    return train(G, P["nodes"], lambda z, ep: ce_targets(z, t), P["seed"])[1]


def heldout(G, P, y, folds=5, seed_off=0):
    nodes = P["nodes"]; rng = np.random.default_rng(P["seed"] + seed_off)
    f = torch.as_tensor(rng.permutation(nodes.numel()) % folds, device=G["device"])
    H = torch.zeros(G["n"], G["C"], device=G["device"])
    for k in range(folds):
        tr, te = nodes[f != k], nodes[f == k]
        t = onehot(y[tr], G["C"])
        H[te] = train(G, tr, lambda z, ep: ce_targets(z, t), P["seed"] * 10 + k)[1][te]
    return H


def rank01(s):
    return torch.argsort(torch.argsort(s)).float() / max(1, s.numel() - 1)


def lp_post(G, P, prob, nodes, y, alpha=0.05, steps=10, clamp=0.1):
    seed = torch.zeros(G["n"], dtype=torch.bool, device=G["device"]); seed[nodes] = True
    lab = torch.zeros(G["n"], dtype=torch.long, device=G["device"]); lab[nodes] = y[nodes]
    return diffuse_probabilities(prob.clamp_min(1e-12).log(), G["e"], alpha=alpha, steps=int(steps),
                                 seed_mask=seed, seed_labels=lab, clamp=clamp)


def recipe_weights(G, P, nodes, y, aw, qw, pw, cbp):
    src = P["src"]; w = torch.ones(nodes.numel(), device=G["device"])
    w[src[nodes] == 1] = aw; w[src[nodes] == 3] = qw; w[src[nodes] == 0] = pw  # 0 = pseudo-label added here
    if cbp > 0:
        cnt = torch.bincount(y[nodes], minlength=G["C"]).float().clamp_min(1)
        cw = cnt[y[nodes]] ** -cbp; w = w * cw / cw.mean()
    return w


class MLP2(torch.nn.Module):
    def __init__(self, d, c, h=64, p=0.5):
        super().__init__()
        self.l1, self.l2, self.p = torch.nn.Linear(d, h), torch.nn.Linear(h, c), p

    def forward(self, x, e=None):
        return self.l2(F.dropout(F.relu(self.l1(x)), self.p, self.training))


def fit_any(G, model, nodes, t, seed, epochs=None):
    epochs = EPOCHS if epochs is None else epochs
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        model.train(); opt.zero_grad(); ce_targets(model(G["x"], G["e"])[nodes], t).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return F.softmax(model(G["x"], G["e"]), 1)


def label_prop(G, nodes, y, alpha=0.9, steps=20):
    """Structure-only label propagation from the given labelled nodes (no node features)."""
    n, C, e = G["n"], G["C"], G["e"]
    Y0 = torch.zeros(n, C, device=G["device"]); Y0[nodes, y] = 1
    deg = torch.zeros(n, device=G["device"]).index_add_(0, e[0], torch.ones(e.shape[1], device=G["device"])).clamp_min(1)
    w = (deg[e[0]] * deg[e[1]]).rsqrt(); Z = Y0.clone()
    for _ in range(steps):
        Z = alpha * torch.zeros_like(Z).index_add_(0, e[1], Z[e[0]] * w[:, None]) + (1 - alpha) * Y0
    s = Z.sum(1, keepdim=True)
    return torch.where(s > 1e-12, Z / s.clamp_min(1e-12), torch.full_like(Z, 1.0 / C))


def difficulty(G, P, mode=0.0):
    """Per-node ease in [0,1]: max class probability of held-out GCNs (mode 0) or of a GCN trained on
    the initial anchors only (mode 1). Uses no information about whether the GNN agrees with the LLM label."""
    v = P["nodes"]
    if mode == 0:
        H = heldout(G, P, P["y"])
    elif mode in (3, 4):  # out-of-fold like mode 2, but text-only MLP (3) or structure-only label propagation (4)
        a = v[P["src"][v] == 1]; rng = np.random.default_rng(P["seed"] + 5)
        f = torch.as_tensor(rng.permutation(a.numel()) % 5, device=G["device"])
        Hs = []; H = torch.zeros(G["n"], G["C"], device=G["device"])
        for k in range(5):
            tr = a[f != k]
            if mode == 3:
                t = onehot(P["y"][tr], G["C"]); m = MLP2(G["x"].shape[1], G["C"]).to(G["device"])
                q = fit_any(G, m, tr, t, P["seed"] * 10 + 77 + k)
            else:
                q = label_prop(G, tr, P["y"][tr])
            Hs.append(q); H[a[f == k]] = q[a[f == k]]
        mean = torch.stack(Hs).mean(0); other = torch.ones(G["n"], dtype=torch.bool, device=G["device"]); other[a] = False
        H[other] = mean[other]
    elif mode == 5:  # as mode 2, but the difficulty reference is always a GCN, whatever the trained backbone
        a = v[P["src"][v] == 1]; rng = np.random.default_rng(P["seed"] + 5)
        f = torch.as_tensor(rng.permutation(a.numel()) % 5, device=G["device"])
        Hs = []; H = torch.zeros(G["n"], G["C"], device=G["device"])
        for k in range(5):
            tr = a[f != k]; t = onehot(P["y"][tr], G["C"])
            torch.manual_seed(P["seed"] * 10 + 77 + k)
            q = fit_any(G, GCN(G["x"].shape[1], G["C"]).to(G["device"]), tr, t, P["seed"] * 10 + 77 + k)
            Hs.append(q); H[a[f == k]] = q[a[f == k]]
        mean = torch.stack(Hs).mean(0); other = torch.ones(G["n"], dtype=torch.bool, device=G["device"]); other[a] = False
        H[other] = mean[other]
    elif mode == 2:  # anchors judged out-of-fold, every other node by the mean of the anchor fold models
        a = v[P["src"][v] == 1]; rng = np.random.default_rng(P["seed"] + 5)
        f = torch.as_tensor(rng.permutation(a.numel()) % 5, device=G["device"])
        Hs = []; H = torch.zeros(G["n"], G["C"], device=G["device"])
        for k in range(5):
            tr = a[f != k]; t = onehot(P["y"][tr], G["C"])
            q = train(G, tr, lambda z, ep: ce_targets(z, t), P["seed"] * 10 + 77 + k)[1]; Hs.append(q)
            H[a[f == k]] = q[a[f == k]]
        mean = torch.stack(Hs).mean(0); other = torch.ones(G["n"], dtype=torch.bool, device=G["device"]); other[a] = False
        H[other] = mean[other]
    else:
        a = v[P["src"][v] == 1]; t = onehot(P["y"][a], G["C"])
        H = train(G, a, lambda z, ep: ce_targets(z, t), P["seed"] + 77)[1]
    return H.max(1).values


def class_preserving(w_old, w_new, y, C):
    """Rescale w_new so that every class keeps the total weight it had under w_old."""
    tot_old = torch.zeros(C, device=w_old.device).index_add_(0, y, w_old)
    tot_new = torch.zeros(C, device=w_old.device).index_add_(0, y, w_new).clamp_min(1e-12)
    return w_new * (tot_old / tot_new)[y]


def m_recipe(G, P, aw=1.5, qw=1.0, pw=0.8, cbp=0.6, lp=1.0, st=0.0, per=1.0, dg=0.0, dmode=0.0, cn=0.0, sf=-1.0):
    """Training recipe of the backbone on the simple pipeline: source weights, class-balanced weights,
    label-propagation post-processing, optional class-balanced self-training (st rounds)."""
    nodes = P["nodes"]; y = P["y"].clone()
    P = dict(P); P["src"] = P["src"].clone()
    ease = difficulty(G, P, dmode) if dg > 0 else None
    sem = None
    if sf >= 0:  # frozen-prototype semantic support of every LLM label (multi-scale), affine weights
        sem = torch.ones(G["n"], device=G["device"])
        sem[nodes] = sf + (1 - sf) * rank01(semantic_support(G, P))
    def fit(nodes, seed):
        w = recipe_weights(G, P, nodes, y, aw, qw, pw, cbp); t = onehot(y[nodes], G["C"])
        if sem is not None:
            llm = (P["src"][nodes] == 1) | (P["src"][nodes] == 3)
            w = torch.where(llm, w * sem[nodes], w)
        if ease is not None:
            llm = (P["src"][nodes] == 1) | (P["src"][nodes] == 3)
            w2 = torch.where(llm, w * ease[nodes].clamp_min(1e-3) ** dg, w)
            w = class_preserving(w, w2, y[nodes], G["C"]) if cn > 0 else w2
        return train(G, nodes, lambda z, ep: ce_targets(z, t, w), seed)[1]
    p = fit(nodes, P["seed"]); cur = nodes
    for r in range(int(st)):
        mask = torch.ones(G["n"], dtype=torch.bool, device=G["device"]); mask[cur] = False
        cand = torch.where(mask)[0]; conf, lab = p[cand].max(1); k = int(per * nodes.numel()) // G["C"]
        pick = torch.cat([cand[lab == c][torch.argsort(conf[lab == c], descending=True)[:k]] for c in range(G["C"])])
        y[pick] = p[pick].argmax(1); P["src"][pick] = 0
        cur = torch.cat([cur, pick]); p = fit(cur, P["seed"] + r + 1)
    if lp > 0:
        p = lp_post(G, P, p, cur, y)
    return p


def semantic_support(G, P, hops=(0, 1, 2), agg="mean"):
    """Frozen-prototype semantic support of each LLM label (the original LENS verifier, simplified):
    class prototypes from the initial anchors; margin of the labelled class over the strongest competitor,
    computed on the node text and on 1-/2-hop neighbourhood means; scales combined by mean or min."""
    v = P["nodes"]; a = v[P["src"][v] == 1]; C = G["C"]; x = G["x"]; e = G["e"]; n = G["n"]
    deg = torch.zeros(n, device=x.device).index_add_(0, e[0], torch.ones(e.shape[1], device=x.device)).clamp_min(1)
    feats = [x]; h = x
    for _ in range(max(hops)):
        h = torch.zeros_like(h).index_add_(0, e[1], h[e[0]]) / deg[:, None]; feats.append(h)
    margins = []
    for k in hops:
        f = F.normalize(feats[k], dim=1)
        proto = F.normalize(torch.zeros(C, x.shape[1], device=x.device).index_add_(0, P["y"][a], f[a]), dim=1)
        s = f[v] @ proto.T; y = P["y"][v]
        lab = s[torch.arange(v.numel()), y]; s2 = s.clone(); s2[torch.arange(v.numel()), y] = -9
        margins.append(lab - s2.max(1).values)
    M = torch.stack(margins)
    return M.mean(0) if agg == "mean" else M.min(0).values


def module_weights(G, P, sf=0.25, dg=0.0):
    """LENS module weights of the training labels: semantic reference x difficulty reference (dmode 5)."""
    v = P["nodes"]; w = sf + (1 - sf) * rank01(semantic_support(G, P))
    if dg > 0:
        w = w * difficulty(G, P, 5)[v].clamp_min(1e-3) ** dg
    llm = (P["src"][v] == 1) | (P["src"][v] == 3)
    return torch.where(llm, w, torch.ones_like(w))


def m_plug(G, P, base=0.0, sf=0.25, dg=0.0):
    """A robust-loss baseline with the LENS module as per-label weights (base 0 GCE, 1 SCE, 2 ELR, 3 Co-teaching)."""
    y = P["y"][P["nodes"]]; nodes = P["nodes"]; C = G["C"]
    w = module_weights(G, P, sf, dg) if sf >= 0 else torch.ones(nodes.numel(), device=G["device"])
    wmean = lambda l, idx=None: (l * w).sum() / w.sum() if idx is None else (l[idx] * w[idx]).sum() / w[idx].sum()
    if int(base) == 3:
        a, b = new_model(G, P["seed"]), new_model(G, P["seed"] + 1000)
        opt = torch.optim.Adam(list(a.parameters()) + list(b.parameters()), lr=0.01, weight_decay=5e-4)
        for ep in range(EPOCHS):
            a.train(); b.train(); opt.zero_grad()
            za, zb = a(G["x"], G["e"])[nodes], b(G["x"], G["e"])[nodes]
            la, lb = F.cross_entropy(za, y, reduction="none"), F.cross_entropy(zb, y, reduction="none")
            keep = int(round((1 - 0.3 * min(1.0, ep / 50)) * y.numel()))
            ia, ib = torch.argsort(la.detach())[:keep], torch.argsort(lb.detach())[:keep]
            (wmean(la, ib) + wmean(lb, ia)).backward(); opt.step()
        a.eval(); b.eval()
        with torch.no_grad():
            return (F.softmax(a(G["x"], G["e"]), 1) + F.softmax(b(G["x"], G["e"]), 1)) / 2
    t = onehot(y, C); ema = torch.zeros(y.numel(), C, device=G["device"])

    def loss(z, ep):
        p = F.softmax(z, 1)
        if int(base) == 0:
            l = (1 - p.gather(1, y[:, None]).squeeze(1).clamp_min(1e-7) ** 0.7) / 0.7
        elif int(base) == 1:
            l = 0.1 * F.cross_entropy(z, y, reduction="none") - (p.clamp(1e-7, 1) * torch.log(t.clamp_min(1e-4))).sum(1)
        else:
            q = p.clamp(1e-4, 1 - 1e-4)
            ema.mul_(0.7).add_(0.3 * (q / q.sum(1, keepdim=True)).detach())
            l = F.cross_entropy(z, y, reduction="none") + 3.0 * torch.log(1 - (ema * q).sum(1))
        return wmean(l)
    return train(G, nodes, loss, P["seed"])[1]


def diffuse_probabilities(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    alpha: float,
    steps: int,
    seed_mask: torch.Tensor | None = None,
    seed_labels: torch.Tensor | None = None,
    clamp: float = 0.0,
    seed_strength: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diffuse predictions while anchoring every step to the model probabilities."""

    if not 0.0 <= alpha <= 1.0 or not 0.0 <= clamp <= 1.0:
        raise ValueError("alpha and clamp must be in [0, 1]")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    base = F.softmax(logits, dim=1)
    current = base.clone()
    source, target = edge_index
    degree = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    degree.index_add_(0, target, torch.ones(target.numel(), dtype=logits.dtype, device=logits.device))
    isolated = degree == 0
    if seed_mask is not None:
        if seed_labels is None:
            raise ValueError("seed_labels are required with seed_mask")
        seed_mask = seed_mask.to(device=logits.device, dtype=torch.bool)
        seed_labels = seed_labels.to(device=logits.device, dtype=torch.long)
        safe_seed_labels = seed_labels.clone()
        safe_seed_labels[~seed_mask] = 0
        if bool(((safe_seed_labels[seed_mask] < 0) | (safe_seed_labels[seed_mask] >= logits.shape[1])).any()):
            raise ValueError("seed labels must be valid class indices inside seed_mask")
        seed_target = F.one_hot(
            safe_seed_labels,
            num_classes=logits.shape[1],
        ).to(logits.dtype)
        if seed_strength is None:
            effective_seed_strength = torch.ones(
                logits.shape[0], dtype=logits.dtype, device=logits.device
            )
        else:
            effective_seed_strength = seed_strength.to(logits.device, logits.dtype).clamp(0.0, 1.0)
    for _ in range(steps):
        neighbor = torch.zeros_like(current)
        neighbor.index_add_(0, target, current[source])
        neighbor = neighbor / degree.clamp_min(1).unsqueeze(1)
        neighbor[isolated] = current[isolated]
        current = (1.0 - alpha) * base + alpha * neighbor
        if seed_mask is not None and clamp > 0:
            anchor = (
                clamp * effective_seed_strength[seed_mask]
            ).unsqueeze(1)
            current[seed_mask] = (
                (1.0 - anchor) * current[seed_mask]
                + anchor * seed_target[seed_mask]
            )
        current = current / current.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return current
