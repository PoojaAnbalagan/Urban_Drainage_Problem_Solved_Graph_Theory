"""Experiment runner. Common random numbers: the same U_e per (network, trial) for all strategies."""
import argparse, os, time
import numpy as np
import pandas as pd
from network import build_network
from cascade import cascade
from centrality import MEASURES, utilisation
from strategies import select
from evaluate import metrics

K_REINF, DELTA = 1.5, 0.5          # capacity multiplier, risk reduction (identical for all strategies)


def calibrate_scale(net, target=0.10, trials=40, seed=12345):
    """Choose capacity margin so Strategy D fails ~target of pipes at theta=1. D only."""
    rng = np.random.default_rng(seed)
    U = rng.random((trials, net.m))
    best, best_gap = None, 9
    for s in np.geomspace(0.1, 8.0, 30):
        net.set_scale(s); rf = []
        for t in range(trials):
            F0 = U[t] < np.minimum(1, net.risk)
            rf.append(cascade(net, net.cap0, net.g, 1.0, F0)["failed"].mean())
        gap = abs(np.mean(rf) - target)
        if gap < best_gap:
            best, best_gap = s, gap
    net.set_scale(best)
    net.calib_gap = best_gap
    return best


def apply_risk_correlation(net, b, rho):
    """True risk = (1-rho)*independent + rho*(same values re-assigned in order of b)."""
    r_ind = net.risk.copy()
    r_sorted = np.sort(r_ind)[np.argsort(np.argsort(b, kind="stable"), kind="stable")]
    net.risk = (1 - rho) * r_ind + rho * r_sorted


def run_network(net_seed, thetas, budgets, trials, lams, n_random=30, sigma=0.0, rho=0.0,
                base_seed=2026, offset=0, scale=None, measure="bet", norm="minmax",
                weight_follows_capacity=True):
    net = build_network(net_seed)
    s = calibrate_scale(net) if scale is None else (net.set_scale(scale) or scale)
    b = MEASURES[measure](net)
    if rho > 0:
        apply_risk_correlation(net, b, rho)
    util = utilisation(net)
    rows = []
    for t in range(trials):
        rng = np.random.default_rng([base_seed, net_seed, offset + t])
        Ue = rng.random(net.m)                                   # common random numbers
        r_obs = net.risk * np.exp(sigma * rng.standard_normal(net.m)) if sigma > 0 else net.risk
        for bud in budgets:
            K = int(np.ceil(bud * net.m))
            sel = {"A_risk": select("A_risk", K, r_obs, b, util, rng),
                   "D_none": select("D_none", K, r_obs, b, util, rng),
                   "E_load": select("E_load", K, r_obs, b, util, rng)}
            for lam in lams:
                sel[f"B_lam{lam:.2f}"] = select("B_graph", K, r_obs, b, util, rng, lam, norm)
            rand = [select("C_random", K, r_obs, b, util, rng) for _ in range(n_random)]
            for theta in thetas:
                def run(idx, name):
                    mask = np.zeros(net.m, bool); mask[idx] = True
                    cap = net.cap0 * np.where(mask, K_REINF, 1.0)
                    w = net.g * np.where(mask, K_REINF, 1.0) if weight_follows_capacity else net.g
                    p = np.minimum(1.0, theta * net.risk * (1 - DELTA * mask))
                    F0 = Ue < p
                    t0 = time.perf_counter()
                    res = cascade(net, cap, w, theta, F0)
                    m = metrics(net, res, F0, theta)
                    m["runtime_ms"] = 1000 * (time.perf_counter() - t0)
                    return m
                for name, idx in sel.items():
                    rows.append(dict(net=net_seed, theta=theta, budget=bud, trial=offset + t,
                                     strategy=name, **run(idx, name)))
                ms = [run(idx, "C") for idx in rand]
                avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
                rows.append(dict(net=net_seed, theta=theta, budget=bud, trial=offset + t,
                                 strategy="C_random", **avg))
    df = pd.DataFrame(rows)
    df["scale"] = s
    df["sigma"], df["rho"] = sigma, rho
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.7, 1.0, 1.3, 1.6])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--measure", default="bet", choices=list(MEASURES))
    ap.add_argument("--split", default="test", choices=["tune", "test"])
    ap.add_argument("--out", default="results/results.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    off = 0 if a.split == "tune" else 1000
    t0 = time.time()
    dfs = [run_network(s, a.thetas, a.budgets, a.trials, a.lams, a.n_random, a.sigma, a.rho,
                       offset=off, measure=a.measure) for s in a.net_seeds]
    pd.concat(dfs).to_csv(a.out, index=False)
    print(f"saved {a.out} in {time.time() - t0:.1f}s")
