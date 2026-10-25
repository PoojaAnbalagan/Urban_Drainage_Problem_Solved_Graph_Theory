"""Priority scores and pipe selection for strategies A-E."""
import numpy as np


def minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x, float)


def rank01(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(len(x) - 1, 1)


def top_k(score, K):
    return np.argsort(-score, kind="stable")[:K]


def priority(r_obs, b, lam, norm="minmax"):
    f = minmax if norm == "minmax" else rank01
    return lam * f(r_obs) + (1 - lam) * f(b)


def select(name, K, r_obs, b, util, rng, lam=None, norm="minmax"):
    if name == "D_none":
        return np.array([], int)
    if name == "A_risk":
        return top_k(r_obs, K)
    if name == "B_graph":
        return top_k(priority(r_obs, b, lam, norm), K)
    if name == "C_random":
        return rng.choice(len(r_obs), K, replace=False)
    if name == "E_load":
        return top_k(util, K)
    raise ValueError(name)
