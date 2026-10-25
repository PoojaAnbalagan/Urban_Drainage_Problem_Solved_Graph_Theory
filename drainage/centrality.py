"""Structural importance measures b_e and comparison measures."""
import networkx as nx
import numpy as np
from cascade import route


def to_digraph(net):
    G = nx.DiGraph()
    G.add_nodes_from(range(net.n))
    for e, (u, v) in enumerate(zip(net.tail_l, net.head_l)):
        G.add_edge(u, v, eid=e, length=float(net.length[e]))
    return G


def edge_betweenness_subset(net):
    """Directed edge betweenness, sources = demand nodes, targets = outlets."""
    G = to_digraph(net)
    src = [v for v in range(net.n) if net.demand0[v] > 0]
    tgt = [v for v in range(net.n) if net.outlets[v]]
    bc = nx.edge_betweenness_centrality_subset(G, sources=src, targets=tgt,
                                               normalized=True, weight="length")
    b = np.zeros(net.m)
    for (u, v), val in bc.items():
        b[G[u][v]["eid"]] = val
    return b


def catchment(net):
    """Demand of all nodes upstream of (and including) the pipe's tail."""
    G = to_digraph(net)
    up = {v: net.demand0[v] + sum(net.demand0[a] for a in nx.ancestors(G, v)) for v in range(net.n)}
    return np.array([up[u] for u in net.tail_l])


def oracle(net):
    """Unmet-demand fraction if this single pipe fails (benchmark, not a deployable score)."""
    out = np.zeros(net.m)
    for e in range(net.m):
        alive = [True] * net.m; alive[e] = False
        _, unmet, _, _ = route(net, alive, net.g.tolist(), 1.0)
        out[e] = unmet / net.demand0.sum()
    return out


def utilisation(net):
    return net.load0 / net.cap0


MEASURES = {"bet": edge_betweenness_subset, "catch": catchment, "oracle": oracle}
