"""Evaluation metrics and paired statistics."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def largest_cut_region(net, reach):
    cut = [v for v in range(net.n) if not reach[v] and not net.outl_l[v]]
    seen, best = set(), 0
    for s in cut:
        if s in seen:
            continue
        stack, size = [s], 0
        seen.add(s)
        while stack:
            v = stack.pop(); size += 1
            for w in net.nbr_l[v]:
                if w not in seen and not reach[w] and not net.outl_l[w]:
                    seen.add(w); stack.append(w)
        best = max(best, size)
    return best


def metrics(net, res, F0, theta):
    failed = res["failed"]
    nf = int(failed.sum())
    tot = theta * net.demand0.sum()
    return dict(Rf=nf / net.m, secondary=nf - int(F0.sum()),
                RC=net.cap0[~failed].sum() / net.cap0.sum(),     # uses original capacities
                U=res["unmet"] / tot, delivered=res["delivered"] / tot,
                region=largest_cut_region(net, res["reach"]), steps=res["steps"])


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def paired_table(df, ref, target, metric, keys=("theta", "budget"), n_boot=2000, seed=0):
    """Paired comparison. diff = ref - target (positive means target is better, lower metric)."""
    piv = df[df.strategy.isin([ref, target])].pivot_table(
        index=list(keys) + ["net", "trial"], columns="strategy", values=metric).reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for k, g in piv.groupby(list(keys)):
        d = (g[ref] - g[target]).to_numpy()
        boots = rng.choice(d, (n_boot, len(d))).mean(1)
        p = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
        mr = g[ref].mean()
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)),
                         mean_ref=mr, mean_target=g[target].mean(), mean_diff=d.mean(),
                         ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                         reduction_pct=100 * d.mean() / mr if mr > 0 else np.nan,
                         p_raw=p, n=len(d)))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_raw.values)
    return out
