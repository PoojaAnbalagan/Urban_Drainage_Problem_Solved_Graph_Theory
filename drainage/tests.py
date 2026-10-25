"""Validation tests (Stage 4 list). Run: python tests.py"""
import numpy as np
import networkx as nx
from network import build_network
from cascade import route, cascade
from centrality import to_digraph, MEASURES, utilisation
from strategies import select
from run_experiments import run_network, calibrate_scale

ok = lambda name, cond, info="": print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}") or cond
res = []
nets = [build_network(s) for s in (101, 102, 103, 104, 105)]
for net in nets:
    calibrate_scale(net)
n = nets[0]

gaps=[x.calib_gap for x in nets]
res.append(ok("0 calibration: D failure rate at theta=1 within 0.02 of 0.10", max(gaps)<0.02, f"gaps={[round(g,3) for g in gaps]}, scales={[round(x.scale,2) for x in nets]}"))
res.append(ok("1 DAG and all nodes reach an outlet",
              all(nx.is_directed_acyclic_graph(to_digraph(x)) for x in nets)
              and all(route(x, [True] * x.m, x.g.tolist(), 1.0)[2].all() for x in nets),
              f"pipes={[x.m for x in nets]}"))
rng = np.random.default_rng(0); worst = 0
for x in nets:
    for _ in range(20):
        alive = (rng.random(x.m) > 0.15).tolist(); th = rng.choice([0.7, 1.3, 1.6])
        _, unmet, _, deliv = route(x, alive, x.g.tolist(), th)
        worst = max(worst, abs(unmet + deliv - th * x.demand0.sum()))
res.append(ok("2 flow conservation (delivered + unmet = demand)", worst < 1e-9, f"max err={worst:.2e}"))
over0 = max((route(x, [True] * x.m, x.g.tolist(), 1.0)[0] > x.cap0 * (1 + 1e-9)).sum() for x in nets)
res.append(ok("3 no overload at theta=1, no failures, no reinforcement", over0 == 0, f"overloads={over0}"))
r = cascade(n, n.cap0, n.g, 1.6, rng.random(n.m) < 0.1, record=True)
mono = all((r["history"][i] <= r["history"][i + 1]).all() for i in range(len(r["history"]) - 1))
res.append(ok("4 cascade terminates, failed set only grows", mono and r["steps"] <= n.m,
              f"steps={r['steps']}"))
b = MEASURES["bet"](n); u = utilisation(n); rr = np.random.default_rng(1)
same = all((select("A_risk", K, n.risk, b, u, rr) == select("B_graph", K, n.risk, b, u, rr, 1.0)).all()
           for K in (5, 9, 18))
res.append(ok("5 B at lambda=1 selects same pipes as A", same))
kw = dict(thetas=[1.3], budgets=[0.1], trials=5, lams=[0.5], n_random=3)
d1, d2 = run_network(101, **kw), run_network(101, **kw)
res.append(ok("6 same seed gives identical results",
              d1.drop(columns="runtime_ms").equals(d2.drop(columns="runtime_ms"))))
d = run_network(101, thetas=[0.7, 1.0, 1.3, 1.6], budgets=[0.10], trials=30, lams=[0.5], n_random=5)
mean = d.groupby(["theta", "strategy"]).Rf.mean().unstack()
print(mean.round(4).to_string())
res.append(ok("7 (soft) mean R_f of D >= A, B, E at theta>=1.3",
              bool((mean.loc[[1.3, 1.6], "D_none"] >= mean.loc[[1.3, 1.6], ["A_risk", "B_lam0.50", "E_load"]].max(axis=1) - 1e-9).all())))
print(f"\n{sum(res)}/{len(res)} passed")
