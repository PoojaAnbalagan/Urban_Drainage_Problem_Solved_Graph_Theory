"""Network, critical-pipe and cascade-example figures."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cascade import cascade
from centrality import MEASURES
from strategies import priority, top_k


def draw(net, ax, width=None, reinforced=None, failed=None, title=""):
    reinforced = np.zeros(net.m, bool) if reinforced is None else reinforced
    failed = np.zeros(net.m, bool) if failed is None else failed
    width = np.full(net.m, 1.2) if width is None else width
    for e, (u, v) in enumerate(zip(net.tail_l, net.head_l)):
        col = "#d62728" if failed[e] else ("#1f77b4" if reinforced[e] else "#8a8a8a")
        ax.annotate("", xy=net.pos[v], xytext=net.pos[u], zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=width[e],
                                    ls="--" if failed[e] else "-", shrinkA=6, shrinkB=6, mutation_scale=9))
    inl = net.demand0 > 0
    ax.scatter(*net.pos[inl].T, s=40, c="#f2a54a", zorder=2, edgecolor="k", lw=.4)
    ax.scatter(*net.pos[~inl & ~net.outlets].T, s=22, c="white", zorder=2, edgecolor="k", lw=.4)
    ax.scatter(*net.pos[net.outlets].T, s=90, marker="s", c="#2ca02c", zorder=3, edgecolor="k")
    ax.set_title(title, fontsize=10); ax.axis("off")


def make_figures(net, outdir, theta=1.6, budget=0.10, seed=3):
    b = MEASURES["bet"](net)
    K = int(np.ceil(budget * net.m))
    S = priority(net.risk, b, 0.5)
    crit = np.zeros(net.m, bool); crit[top_k(S, K)] = True
    fig, ax = plt.subplots(figsize=(7, 6.5))
    draw(net, ax, title="Synthetic drainage network (orange: inlets, green squares: outlets)")
    fig.savefig(f"{outdir}/fig1_network.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    draw(net, ax, width=0.8 + 4 * S / S.max(), reinforced=crit,
         title=f"Strategy B (lambda=0.5): top {K} pipes (blue); width = priority score")
    fig.savefig(f"{outdir}/fig2_critical_pipes.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    rng = np.random.default_rng(seed); best = None
    for _ in range(200):                                  # pick an illustrative multi-step cascade
        F0 = rng.random(net.m) < np.minimum(1, theta * net.risk)
        r = cascade(net, net.cap0, net.g, theta, F0, record=True)
        if best is None or r["steps"] > best["steps"]:
            best = r
    h = best["history"][:4]
    fig, axs = plt.subplots(1, len(h), figsize=(4 * len(h), 4.2))
    for i, (a, f) in enumerate(zip(np.atleast_1d(axs), h)):
        draw(net, a, failed=f, title=f"{'Initial failures' if i == 0 else f'Round {i}'}: {int(f.sum())} failed")
    fig.suptitle(f"Failure propagation example (theta={theta}, no intervention)")
    fig.savefig(f"{outdir}/fig3_cascade_example.png", dpi=170, bbox_inches="tight"); plt.close(fig)
