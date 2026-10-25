"""Stage 6 analysis: tables + colour-blind-safe figures with data-driven takeaway titles.
Palette: Okabe-Ito. Run after run_experiments.py has produced results/*.csv."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr
from evaluate import paired_table
from network import build_network
from centrality import MEASURES, utilisation
from run_experiments import calibrate_scale
from strategies import priority, top_k

R, FIG, TAB = "results", "results/figs", "results/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
PAL = {"A_risk": "#0072B2", "B": "#D55E00", "C_random": "#8C8C8C", "D_none": "#222222", "E_load": "#009E73"}
MRK = {"A_risk": "o", "B": "s", "C_random": "^", "D_none": "X", "E_load": "D"}
ORDER = ["D_none", "C_random", "A_risk", "E_load", "B"]
GOOD, TIE, BAD, MIX = "#009E73", "#BDBDBD", "#D55E00", "#E69F00"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "figure.dpi": 110, "savefig.dpi": 200})

tune, test = pd.read_csv(f"{R}/tune.csv"), pd.read_csv(f"{R}/test.csv")
bn = [s for s in tune.strategy.unique() if s.startswith("B_")]
LS = tune[tune.strategy.isin(bn)].groupby("strategy").Rf.mean().idxmin()      # lambda* from TUNE only
LAM = float(LS.split("lam")[1])
LAB = {"A_risk": "A  Risk-based", "B": f"B  Graph-based (λ*={LAM:g})", "C_random": "C  Random",
       "D_none": "D  No intervention", "E_load": "E  Load-based"}
THETAS, BUDS = sorted(test.theta.unique()), sorted(test.budget.unique())


def tag(df):
    d = df.copy(); d["S"] = np.where(d.strategy == LS, "B", d.strategy)
    return d[d.S.isin(ORDER)]


T = tag(test)


def summ(df, metric, by):
    g = df.groupby(by)[metric].agg(["mean", "std", "count"]).reset_index()
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"]); return g


def head(fig, title, sub, foot):
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.925, sub, ha="left", fontsize=9.5, color="#444")
    fig.text(0.01, 0.005, foot, ha="left", fontsize=8.6, color="#555", style="italic")


def legend(fig, y=0.885):
    hs = [plt.Line2D([], [], color=PAL[s], marker=MRK[s], lw=2, label=LAB[s]) for s in ORDER[::-1]]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.01, y), ncol=5, frameon=False, fontsize=9)


# ---------- paired statistics (B* vs A, B* vs E) ----------
PA = {m: paired_table(T.assign(strategy=T.S), "A_risk", "B", m) for m in ["Rf", "U", "secondary", "region"]}
PE = {m: paired_table(T.assign(strategy=T.S), "E_load", "B", m) for m in ["Rf", "U"]}
PA["RC"] = paired_table(T.assign(strategy=T.S), "B", "A_risk", "RC")   # RC: higher is better, so ref=B
for k, v in {**{f"B_vs_A_{m}": PA[m] for m in PA}, **{f"B_vs_E_{m}": PE[m] for m in PE}}.items():
    v.round(5).to_csv(f"{TAB}/paired_{k}.csv", index=False)


def classify(p):
    sig = p.p_holm < .05
    return int((sig & (p.mean_diff > 0)).sum()), int((sig & (p.mean_diff < 0)).sum()), len(p)


def phrase(p, what="R_f"):
    b, w, n = classify(p)
    if b == 0 and w == 0: return f"B* shows no significant {what} difference from A in any of {n} cells"
    if w == 0: return f"B* significantly lowers {what} vs A in {b} of {n} cells, never worse"
    return f"B* lowers {what} in {b} cells but is worse in {w} of {n}"


# ---------- tables ----------
t1 = T[T.budget == .10].groupby(["theta", "S"])[["Rf", "U", "RC", "secondary", "region", "runtime_ms"]].agg(["mean", "std"]).round(4)
t1.to_csv(f"{TAB}/T1_means_budget10.csv")
T[["theta", "budget", "S", "Rf", "U", "RC"]].groupby(["theta", "budget", "S"]).mean().round(4).unstack("S").to_csv(f"{TAB}/T2_all_cells_means.csv")

# ---------- Fig 4: headline ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 5.4)); w = .16
for ax, m, nm in zip(axs, ["Rf", "U"], ["Failed pipes (R_f)", "Unmet demand (U)"]):
    s = summ(T[T.budget == .10], m, ["theta", "S"])
    for i, st in enumerate(ORDER):
        q = s[s.S == st].sort_values("theta")
        ax.bar(np.arange(len(THETAS)) + (i - 2) * w, q["mean"], w, yerr=q.ci, capsize=2, color=PAL[st],
               edgecolor="white", label=LAB[st], error_kw=dict(lw=1, ecolor="#333"))
    pp = PA[m][PA[m].budget == .10].sort_values("theta")
    ax.set_xticks(range(len(THETAS)))
    ax.set_xticklabels([f"θ={t}\nB vs A: {r:+.1f}%{'*' if p < .05 else ''}" for t, r, p in zip(pp.theta, pp.reduction_pct, pp.p_holm)], fontsize=9)
    ax.set_ylabel(nm + "  (lower is better)"); ax.set_title(nm, fontsize=11)
p10R, p10U = PA["Rf"][PA["Rf"].budget == .10], PA["U"][PA["Rf"].budget == .10]
bR, wR, nR = classify(p10R); bUU, wUU, _ = classify(p10U)
head(fig, f"At a 10% budget, B vs A: failed pipes {'significantly lower' if bR else 'not significantly different'} ({bR}/{nR} rainfall levels); unmet demand significantly lower in {bUU}/{nR}",
     "Mean over 5 networks x 100 test trials; error bars = 95% CI. Under each θ: % reduction of B relative to A (negative = B worse); * = Holm-adjusted p < 0.05.",
     "How to read: shorter bars are better. D (black) is the no-maintenance reference; C (grey) is chance.")
legend(fig); fig.tight_layout(rect=[0, .03, 1, .86]); fig.savefig(f"{FIG}/fig4_headline_comparison.png"); plt.close(fig)

# ---------- Fig 5: rainfall response ----------
fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for r, (m, nm) in enumerate([("Rf", "Failed pipes (R_f)"), ("U", "Unmet demand (U)")]):
    for c, bud in enumerate(BUDS):
        ax = axs[r, c]; s = summ(T[T.budget == bud], m, ["theta", "S"])
        for st in ORDER:
            q = s[s.S == st].sort_values("theta")
            ax.plot(q.theta, q["mean"], color=PAL[st], marker=MRK[st], lw=2, ms=5)
            ax.fill_between(q.theta, q["mean"] - q.ci, q["mean"] + q.ci, color=PAL[st], alpha=.18, lw=0)
        ax.set_title(f"Budget {int(bud * 100)}%", fontsize=10); ax.set_xticks(THETAS)
        if c == 0: ax.set_ylabel(nm)
        if r == 1: ax.set_xlabel("Rainfall multiplier θ")
head(fig, "Damage rises steeply with rainfall; the gaps between strategies are small compared with that rise",
     "Mean ± 95% CI over 5 networks x 100 trials. Top row: share of pipes failed. Bottom row: share of demand that cannot reach an outlet.",
     "How to read: curves that stay lower resist rainfall better. Overlapping bands mean no clear difference.")
legend(fig); fig.tight_layout(rect=[0, .03, 1, .86]); fig.savefig(f"{FIG}/fig5_rainfall_response.png"); plt.close(fig)

# ---------- Fig 6: budget response ----------
fig, axs = plt.subplots(2, 2, figsize=(11.5, 7), sharex=True)
for r, (m, nm) in enumerate([("Rf", "Failed pipes (R_f)"), ("U", "Unmet demand (U)")]):
    for c, th in enumerate([1.3, 1.6]):
        ax = axs[r, c]; s = summ(T[T.theta == th], m, ["budget", "S"])
        for st in ORDER:
            q = s[s.S == st].sort_values("budget")
            ax.errorbar(q.budget * 100, q["mean"], yerr=q.ci, color=PAL[st], marker=MRK[st], lw=2, ms=5, capsize=2)
        ax.set_xticks([b * 100 for b in BUDS]); ax.set_title(f"θ = {th}", fontsize=10)
        if c == 0: ax.set_ylabel(nm)
        if r == 1: ax.set_xlabel("Reinforced pipes (% of network)")
gain = {st: T[(T.theta == 1.6) & (T.S == st) & (T.budget == .20)].Rf.mean() for st in ["D_none", "A_risk", "B"]}
head(fig, f"Doubling the budget helps: at θ=1.6, R_f falls from {gain['D_none']:.3f} (none) to {gain['A_risk']:.3f} (A) and {gain['B']:.3f} (B) at 20%",
     "Mean ± 95% CI over 5 networks x 100 trials; strategy D does not use the budget, so it is flat by design.",
     "How to read: steeper downward lines mean extra budget is worth more. Lines that converge mean strategy choice matters less at large budgets.")
legend(fig); fig.tight_layout(rect=[0, .03, 1, .86]); fig.savefig(f"{FIG}/fig6_budget_response.png"); plt.close(fig)

# ---------- Fig 7: lambda sweep (tune set) ----------
tt = tune[tune.budget == .10].copy()
fig, axs = plt.subplots(1, 2, figsize=(12, 5)); cols = plt.cm.viridis(np.linspace(.05, .9, len(THETAS)))
lams = sorted(float(s.split("lam")[1]) for s in bn) + [1.0]
for ax, m, nm in zip(axs, ["Rf", "U"], ["Failed pipes (R_f)", "Unmet demand (U)"]):
    for th, col in zip(THETAS, cols):
        y = [tt[(tt.strategy == f"B_lam{l:.2f}") & (tt.theta == th)][m].mean() for l in lams[:-1]]
        y.append(tt[(tt.strategy == "A_risk") & (tt.theta == th)][m].mean())
        ax.plot(lams, y, marker="o", lw=2, color=col, label=f"θ={th}")
    ax.axvline(LAM, color=PAL["B"], ls="--", lw=1.5); ax.text(LAM, ax.get_ylim()[1], " λ* (chosen)", color=PAL["B"], va="top", fontsize=9)
    ax.set_xlabel("λ  (0 = structure only, 1 = risk only = Strategy A)"); ax.set_ylabel(nm); ax.set_xticks(lams)
ax.legend(frameon=False, fontsize=9)
head(fig, f"λ sweep on the tuning set: lowest mean R_f at λ = {LAM:g}; the curve is {'nearly flat' if np.ptp([tt[tt.strategy == s].Rf.mean() for s in bn]) < .01 else 'not flat'}",
     "Tuning seeds only (trials 0-99). λ* was chosen here and then fixed for the test comparison.",
     "How to read: left end = importance only, right end = Strategy A. A U-shape would mean a blend beats both extremes.")
fig.tight_layout(rect=[0, .03, 1, .88]); fig.savefig(f"{FIG}/fig7_lambda_sweep.png"); plt.close(fig)

# ---------- Fig 8: heatmaps of reduction vs A ----------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
for ax, m, nm in zip(axs, ["Rf", "U"], ["R_f", "U"]):
    p = PA[m]; M = p.pivot(index="budget", columns="theta", values="reduction_pct")
    S = p.pivot(index="budget", columns="theta", values="p_holm")
    v = np.nanmax(np.abs(M.values)) or 1
    im = ax.imshow(M.values, cmap="PuOr", vmin=-v, vmax=v, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels([f"θ={c}" for c in M.columns])
    ax.set_yticks(range(len(M.index))); ax.set_yticklabels([f"{int(i * 100)}%" for i in M.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M.values[i, j]
            ax.text(j, i, "n/a" if np.isnan(val) else f"{val:+.1f}%{'*' if S.values[i, j] < .05 else ''}", ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if (not np.isnan(val) and abs(val) / v > .5) else "#222")
    ax.set_title(f"Reduction in {nm} vs risk-only (A)", fontsize=10.5); ax.set_ylabel("Budget")
    fig.colorbar(im, ax=ax, shrink=.85, label="% reduction (+ = B better)")
head(fig, f"Where does B beat A? Failed pipes: significant in {classify(PA['Rf'])[0]}/12 cells. Unmet demand: significant in {classify(PA['U'])[0]}/12 cells, worse in {classify(PA['U'])[1]}",
     "Percent reduction of the mean, B (λ*) relative to A, paired by trial. * = Holm-adjusted Wilcoxon p < 0.05 across the 12 cells.",
     "How to read: purple = graph-based is better, orange = graph-based is worse, pale = little difference. Darker = larger effect.")
fig.tight_layout(rect=[0, .03, 1, .86]); fig.savefig(f"{FIG}/fig8_reduction_heatmap.png"); plt.close(fig)

# ---------- Fig 9: win / tie / loss ----------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
wtl = {}
for ax, m, nm in zip(axs, ["Rf", "U"], ["R_f", "U"]):
    piv = T[T.budget == .10].pivot_table(index=["theta", "net", "trial"], columns="S", values=m).reset_index()
    rows = []
    for th, g in piv.groupby("theta"):
        d = (g["A_risk"] - g["B"]).to_numpy()
        rows.append((th, 100 * (d > 1e-12).mean(), 100 * (np.abs(d) <= 1e-12).mean(), 100 * (d < -1e-12).mean()))
    q = pd.DataFrame(rows, columns=["theta", "better", "tie", "worse"]); wtl[m] = q
    left = np.zeros(len(q))
    for col, c, lab, h in [("better", GOOD, "B better", ""), ("tie", TIE, "Tie", "//"), ("worse", BAD, "B worse", "xx")]:
        ax.barh(range(len(q)), q[col], left=left, color=c, hatch=h, edgecolor="white", label=lab)
        for i, (l, v) in enumerate(zip(left, q[col])):
            if v > 6: ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white" if col != "tie" else "#222", fontweight="bold", fontsize=9)
        left += q[col].values
    ax.set_yticks(range(len(q))); ax.set_yticklabels([f"θ={t}" for t in q.theta]); ax.set_xlim(0, 100); ax.set_xlabel("% of trials (10% budget)")
    ax.set_title(f"B vs A trial by trial: {nm}", fontsize=10.5); ax.grid(False)
axs[0].legend(ncol=3, frameon=False, bbox_to_anchor=(0, 1.0), loc="lower left", fontsize=9) if False else axs[1].legend(ncol=3, frameon=False, loc="lower right", fontsize=8.5)
tb = wtl["Rf"].better.mean(); tw = wtl["Rf"].worse.mean(); tt_ = wtl["Rf"].tie.mean()
head(fig, f"Consistency check (R_f, 10% budget): B beats A in {tb:.0f}% of trials, ties in {tt_:.0f}%, loses in {tw:.0f}%",
     "Each trial applies the same failure draws to both strategies, so a tie means the two reinforced sets led to identical outcomes.",
     "How to read: green share larger than orange share = B tends to win. A large grey share means the choice often made no difference.")
fig.tight_layout(rect=[0, .03, 1, .86]); fig.savefig(f"{FIG}/fig9_win_tie_loss.png"); plt.close(fig)

# ---------- Fig 10: stress tests ----------
def pooled(df, label, fam):
    d = tag(df); d = d[(d.theta >= 1.0) & (d.budget == .10)]
    piv = d.pivot_table(index=["theta", "net", "trial"], columns="S", values="Rf")
    x = (piv["A_risk"] - piv["B"]).to_numpy(); ma = piv["A_risk"].mean()
    bt = np.random.default_rng(0).choice(x, (2000, len(x))).mean(1)
    return dict(condition=label, family=fam, red=100 * x.mean() / ma, lo=100 * np.percentile(bt, 2.5) / ma, hi=100 * np.percentile(bt, 97.5) / ma)
S10 = pd.DataFrame([pooled(test, "Base: betweenness, accurate risk", "Base"),
                    pooled(pd.read_csv(f"{R}/catch.csv"), "Importance = upstream catchment", "Alt. importance measure"),
                    pooled(pd.read_csv(f"{R}/oracle.csv"), "Importance = oracle (single-pipe impact)", "Alt. importance measure"),
                    pooled(pd.read_csv(f"{R}/noise05.csv"), "Noisy risk estimates (σ=0.5)", "Noisy risk"),
                    pooled(pd.read_csv(f"{R}/noise10.csv"), "Noisy risk estimates (σ=1.0)", "Noisy risk"),
                    pooled(pd.read_csv(f"{R}/corr05.csv"), "Risk correlated with structure (ρ=0.5)", "Correlated risk"),
                    pooled(pd.read_csv(f"{R}/corr10.csv"), "Risk correlated with structure (ρ=1.0)", "Correlated risk")])
S10.round(3).to_csv(f"{TAB}/T4_stress_tests.csv", index=False)
FC = {"Base": PAL["A_risk"], "Alt. importance measure": PAL["E_load"], "Noisy risk": MIX, "Correlated risk": "#CC79A7"}
fig, ax = plt.subplots(figsize=(12, 5.4)); y = np.arange(len(S10))[::-1]
ax.barh(y, S10.red, color=[FC[f] for f in S10.family], height=.62)
ax.errorbar(S10.red, y, xerr=[S10.red - S10.lo, S10.hi - S10.red], fmt="none", ecolor="#222", capsize=3, lw=1.3)
ax.axvline(0, color="#222", lw=1); ax.set_yticks(y); ax.set_yticklabels(S10.condition)
xr = S10.hi.max() + .35; ax.set_xlim(S10.lo.min() - .5, xr + 1.3)
for yy, r, lo_, hi_ in zip(y, S10.red, S10.lo, S10.hi):
    lab = "identical picks (B = A)" if abs(hi_ - lo_) < 1e-9 else f"{r:+.1f}%  [{lo_:+.1f}, {hi_:+.1f}]"
    ax.text(xr, yy, lab, va="center", ha="left", fontsize=9, fontweight="bold")
ax.set_xlabel("Reduction in R_f of B vs A (%, positive = B better)"); ax.grid(axis="y", alpha=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=f) for f, c in FC.items()], frameon=False, loc="lower left", fontsize=8.5, ncol=2)
clear = S10[S10.lo > 0]
head(fig, f"Stress tests: B's R_f gain is clearly positive only under {', '.join(sorted(set(clear.family.str.lower()))) or 'no condition'}",
     f"Range {S10.red.min():+.1f}% to {S10.red.max():+.1f}%. Pooled over θ ≥ 1.0, 10% budget, 5 networks x 100 test trials; whiskers = bootstrap 95% CI of the paired difference.",
     "How to read: a bar is clearly different from risk-only when its whisker does not cross zero. At ρ=1.0 risk and structure rank pipes identically, so B and A coincide by construction.")
fig.tight_layout(rect=[0, .03, 1, .88]); fig.savefig(f"{FIG}/fig10_stress_tests.png"); plt.close(fig)

# ---------- Fig 11: proxy quality (RQ4) ----------
names = ["risk r", "betweenness b", "catchment", "load L/C", "oracle impact"]; mats, cap = [], {"A_risk": [], "B": [], "E_load": []}
for sd in [101, 102, 103, 104, 105]:
    net = build_network(sd); calibrate_scale(net)
    X = np.c_[net.risk, MEASURES["bet"](net), MEASURES["catch"](net), utilisation(net), MEASURES["oracle"](net)]
    mats.append(spearmanr(X).statistic)
    K = int(np.ceil(.10 * net.m)); orc = set(top_k(X[:, 4], K))
    sel = {"A_risk": top_k(X[:, 0], K), "B": top_k(priority(X[:, 0], X[:, 1], LAM), K), "E_load": top_k(X[:, 3], K)}
    for k, v in sel.items(): cap[k].append(len(orc & set(v)) / K)
Mm = np.mean(mats, 0); pd.DataFrame(Mm, index=names, columns=names).round(3).to_csv(f"{TAB}/T5_spearman.csv")
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw=dict(width_ratios=[1.1, 1]))
im = axs[0].imshow(Mm, cmap="PuOr", vmin=-1, vmax=1); axs[0].grid(False)
axs[0].set_xticks(range(5)); axs[0].set_xticklabels(names, rotation=30, ha="right"); axs[0].set_yticks(range(5)); axs[0].set_yticklabels(names)
for i in range(5):
    for j in range(5): axs[0].text(j, i, f"{Mm[i, j]:.2f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white" if abs(Mm[i, j]) > .5 else "#222")
fig.colorbar(im, ax=axs[0], shrink=.8, label="Spearman ρ"); axs[0].set_title("How related are the candidate scores?", fontsize=10.5)
ks = ["A_risk", "E_load", "B"]; vals = [np.mean(cap[k]) * 100 for k in ks]
axs[1].bar([LAB[k] for k in ks], vals, color=[PAL[k] for k in ks]); axs[1].axhline(10, color="#555", ls="--"); axs[1].text(-.45, 10.4, "chance = 10%", ha="left", fontsize=9, color="#555")
for i, v in enumerate(vals): axs[1].text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
axs[1].set_ylabel("% of the true top-10% critical pipes captured"); axs[1].set_title("Which score finds the pipes that matter most?", fontsize=10.5)
plt.setp(axs[1].get_xticklabels(), fontsize=8.5)
head(fig, f"Betweenness vs true impact: ρ = {Mm[1, 4]:.2f}; load vs true impact: ρ = {Mm[3, 4]:.2f}; risk vs true impact: ρ = {Mm[0, 4]:.2f}",
     "Spearman correlation pooled over the 5 networks (left). Right: overlap of each strategy's selected pipes with the oracle top-10% (single-pipe unmet demand).",
     "How to read: oracle impact is a benchmark, not a deployable score. A higher correlation means a better proxy for importance under this model's routing rule (not real hydraulics).")
fig.tight_layout(rect=[0, .03, 1, .86]); fig.savefig(f"{FIG}/fig11_proxy_quality.png"); plt.close(fig)

# ---------- Fig 0: scorecard ----------
bA, wA, nA = classify(PA["Rf"]); bU, wU, _ = classify(PA["U"]); bE, wE, _ = classify(PE["Rf"])
rcm = T.groupby("S").RC.mean()
best_cell = PA["Rf"].loc[PA["Rf"].reduction_pct.idxmax()]
worst_cell = PA["Rf"].loc[PA["Rf"].reduction_pct.idxmin()]
def chip(b, w, n):
    return ("SUPPORTED", GOOD) if b >= .75 * n and w == 0 else (("NOT SUPPORTED", BAD) if b == 0 else ("MIXED", MIX))
rows = [("RQ1", "Does B reduce the share of failed pipes?", f"Significant gain vs A in {bA}/{nA} cells; significantly worse in {wA}. Largest: {best_cell.reduction_pct:+.1f}% (θ={best_cell.theta}, {int(best_cell.budget * 100)}%).", chip(bA, wA, nA)),
        ("RQ2", "Does B keep more capacity and deliver more water?", f"U: significant gain in {bU}/{nA} cells, worse in {wU}. Mean R_C: A {rcm['A_risk']:.3f}, B {rcm['B']:.3f}.", chip(bU, wU, nA)),
        ("RQ3", "How do rainfall and budget change the result?", f"Cell range of R_f change vs A: {worst_cell.reduction_pct:+.1f}% (θ={worst_cell.theta}, {int(worst_cell.budget * 100)}%) to {best_cell.reduction_pct:+.1f}%.", ("SEE FIG 5, 6, 8", "#0072B2")),
        ("RQ4", "Which graph property finds critical pipes?", f"ρ with oracle impact: betweenness {Mm[1, 4]:.2f}, catchment {Mm[2, 4]:.2f}, load {Mm[3, 4]:.2f}, risk {Mm[0, 4]:.2f}.", ("SEE FIG 11", "#0072B2")),
        ("RQ5", "When does B fail to beat A?", f"B vs E (load): gain in {bE}/{nA} cells. Stress tests range {S10.red.min():+.1f}% to {S10.red.max():+.1f}% (Fig 10).", ("SEE FIG 8, 10", "#0072B2"))]
fig = plt.figure(figsize=(13, 6.3)); ax = fig.add_axes([0.01, 0.05, 0.98, 0.84]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for i, (rq, q, res, (lab, col)) in enumerate(rows):
    y0 = .98 - (i + 1) * .19
    ax.add_patch(FancyBboxPatch((0, y0), .99, .165, boxstyle="round,pad=0.005,rounding_size=0.015", fc="#F4F6F8", ec="#D0D5DA"))
    ax.add_patch(FancyBboxPatch((.01, y0 + .03), .06, .105, boxstyle="round,pad=0.003,rounding_size=0.01", fc="#222", ec="none"))
    ax.text(.04, y0 + .082, rq, color="white", ha="center", va="center", fontweight="bold", fontsize=11)
    ax.text(.09, y0 + .115, q, fontweight="bold", fontsize=11, va="center"); ax.text(.09, y0 + .05, res, fontsize=9.6, va="center", color="#333")
    ax.add_patch(FancyBboxPatch((.80, y0 + .045, ), .185, .075, boxstyle="round,pad=0.003,rounding_size=0.02", fc=col, ec="none"))
    ax.text(.8925, y0 + .0825, lab, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
fig.suptitle("Findings at a glance (test set, 5 networks x 100 trials, B uses λ* chosen on tuning data)", x=.01, ha="left", fontsize=13.5, fontweight="bold")
fig.text(.01, .005, "Verdict rule: SUPPORTED = significant gain in >=75% of cells and none worse; NOT SUPPORTED = no significant gain; MIXED otherwise. Simulation on a synthetic network; not real-world flood prevention.", fontsize=8.4, color="#555", style="italic")
fig.savefig(f"{FIG}/fig0_scorecard.png"); plt.close(fig)
print("lambda* =", LAM); print(PA["Rf"].round(4).to_string(index=False)); print(PA["U"].round(4)[["theta", "budget", "mean_ref", "mean_target", "reduction_pct", "p_holm"]].to_string(index=False))
print(S10.round(2).to_string(index=False)); print(pd.DataFrame(Mm, index=names, columns=names).round(2)); print({k: round(np.mean(v), 3) for k, v in cap.items()})
print(wtl["Rf"].round(1).to_string(index=False))
