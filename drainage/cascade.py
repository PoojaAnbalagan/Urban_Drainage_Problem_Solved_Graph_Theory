"""Flow routing and iterative cascading-failure engine.

APPROXIMATION: steady-state routing on a gravity-oriented DAG. Not a hydraulic simulator.
Rule: a node sends its total inflow q_v only along surviving pipes whose head can still
reach an outlet, split in proportion to the pipes' routing weights w_e. Capacity is NOT
enforced during routing, so L_e > C_e (overload) can occur and triggers failure.
"""
import numpy as np


def route(net, alive, w, theta):
    """Return (load per pipe, unmet demand, reach mask, delivered flow)."""
    head, out, outl, dem = net.head_l, net.out_l, net.outl_l, net.dem_l
    reach = outl[:]
    for v in net.order_rev:                      # low elevation first
        if not reach[v]:
            for e in out[v]:
                if alive[e] and reach[head[e]]:
                    reach[v] = True
                    break
    inflow = [0.0] * net.n
    load = [0.0] * net.m
    unmet = delivered = 0.0
    for v in net.order:                          # high elevation first
        q = theta * dem[v] + inflow[v]
        if outl[v]:
            delivered += q
            continue
        if not reach[v]:
            unmet += theta * dem[v]
            continue
        elig = [e for e in out[v] if alive[e] and reach[head[e]]]
        tot = sum(w[e] for e in elig)
        for e in elig:
            f = q * w[e] / tot
            load[e] = f
            inflow[head[e]] += f
    return np.array(load), unmet, np.array(reach), delivered


def cascade(net, cap, w, theta, F0, record=False):
    """Iterate: route -> mark overloaded pipes failed -> repeat until stable.
    Failures are simultaneous within a round. F only grows, so it terminates."""
    failed = F0.copy()
    history = [failed.copy()] if record else None
    steps = 0
    wl = list(w)
    while True:
        alive = (~failed).tolist()
        load, unmet, reach, delivered = route(net, alive, wl, theta)
        over = (~failed) & (load > cap * (1 + 1e-9))
        if not over.any():
            break
        failed |= over
        steps += 1
        if record:
            history.append(failed.copy())
    return dict(failed=failed, load=load, unmet=unmet, reach=reach,
                delivered=delivered, steps=steps, history=history)
