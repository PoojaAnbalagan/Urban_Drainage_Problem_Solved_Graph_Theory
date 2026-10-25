"""Synthetic drainage network generator (units: L/s, m).
NOT a validated representation of any real city's drainage infrastructure."""
import numpy as np
from cascade import route


class Network:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.m = len(self.tail)
        self.tail_l, self.head_l = self.tail.tolist(), self.head.tolist()
        self.outl_l = self.outlets.tolist()
        self.dem_l = self.demand0.tolist()
        self.order = np.argsort(-self.z).tolist()      # edges go strictly downhill
        self.order_rev = self.order[::-1]
        self.out_l = [[] for _ in range(self.n)]
        for e, (u, v) in enumerate(zip(self.tail_l, self.head_l)):
            self.out_l[u].append(e)
        nb = [set() for _ in range(self.n)]
        for u, v in zip(self.tail_l, self.head_l):
            nb[u].add(v); nb[v].add(u)
        self.nbr_l = [sorted(s) for s in nb]

    def set_scale(self, s):
        """Capacity C_e = max((1 + s*alpha_e) * L0_e, cmin); alpha_e ~ U(0.5, 1.5)."""
        self.scale = s
        self.cap0 = np.maximum((1 + s * self.alpha_raw) * self.load0, self.cmin)


def build_network(seed, rows=7, cols=8, n_outlets=3, thin=0.10, inlet_frac=0.6,
                  scale=0.3, cmin=0.5):
    rng = np.random.default_rng(seed)
    n = rows * cols
    r_idx, c_idx = np.arange(n) // cols, np.arange(n) % cols
    pos = np.c_[c_idx, -r_idx].astype(float)
    z = (rows - 1 - r_idx) * 10.0 + rng.normal(0, 3, n)
    outl = rng.choice(np.arange((rows - 1) * cols, n), n_outlets, replace=False)
    outlets = np.zeros(n, bool); outlets[outl] = True
    z[outl] -= 15
    adj = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
    adj += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
    nbrs = [[] for _ in range(n)]
    for a, b in adj:
        nbrs[a].append(b); nbrs[b].append(a)
    for _ in range(200):                                # fill pits so every node drains
        changed = False
        for v in range(n):
            if not outlets[v] and all(z[w] >= z[v] for w in nbrs[v]):
                z[v] = min(z[w] for w in nbrs[v]) + rng.uniform(0.1, 1.0)
                changed = True
        if not changed:
            break
    edges = []
    for a, b in adj:
        u, v = (a, b) if z[a] > z[b] else (b, a)
        if not outlets[u]:
            edges.append((u, v))
    out_deg = np.bincount([u for u, _ in edges], minlength=n)
    remove, target = set(), int(thin * len(edges))
    for i in rng.permutation(len(edges)):               # thin, keeping out-degree >= 1
        if len(remove) >= target:
            break
        u = edges[i][0]
        if out_deg[u] > 1:
            remove.add(i); out_deg[u] -= 1
    edges = [e for i, e in enumerate(edges) if i not in remove]
    tail = np.array([u for u, _ in edges]); head = np.array([v for _, v in edges])
    m = len(edges)
    in_deg = np.bincount(head, minlength=n)
    inlet = ((rng.random(n) < inlet_frac) | (in_deg == 0)) & ~outlets
    demand0 = np.where(inlet, 2.0 * rng.lognormal(0, 0.5, n), 0.0)
    net = Network(seed=seed, n=n, pos=pos, z=z, outlets=outlets, demand0=demand0,
                  tail=tail, head=head, length=rng.uniform(50, 150, m),
                  g=rng.lognormal(0, 0.4, m), alpha_raw=rng.uniform(0.5, 1.5, m),
                  risk=rng.beta(2, 30, m), cmin=cmin)
    load, unmet, reach, _ = route(net, [True] * m, net.g.tolist(), 1.0)
    assert unmet == 0 and reach.all(), "every node must reach an outlet"
    net.load0 = load
    net.set_scale(scale)
    return net
