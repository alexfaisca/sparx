"""Every algorithm on degenerate graphs (no nodes, one node, nodes without edges, a single edge),
undirected and directed, each with its exact result: what they return there is part of the API."""

import gc
import math
import os

import numpy as np
import pytest

import sparx


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    """Runs each test in its own directory (the cache is `./.cache/`), and checks it's left empty."""
    monkeypatch.chdir(tmp_path)
    yield
    gc.collect()
    left = sorted(os.listdir(tmp_path / ".cache")) if (tmp_path / ".cache").exists() else []
    assert left == [], f"cache files left behind: {left}"


def load(n, arcs, directed, weighted=False):
    """A graph of `n` nodes: undirected, `arcs` are its edges (each once), else its arcs."""
    kind = "general" if directed else "symmetric"
    field = "real" if weighted else "pattern"
    with open("g.mtx", "w") as f:
        f.write(f"%%MatrixMarket matrix coordinate {field} {kind}\n{n} {n} {len(arcs)}\n")
        for u, v in arcs:
            # symmetric files hold each edge once, from the larger node
            u, v = (max(u, v), min(u, v)) if not directed else (u, v)
            f.write(f"{u + 1} {v + 1}" + (" 2.5" if weighted else "") + "\n")
    cls = sparx.DiGraph if directed else sparx.Graph
    return cls("g.mtx", index="u32", threads=4, weighted=weighted)


def eq(a, b):
    assert np.asarray(a).tolist() == list(b), (a, b)


# no nodes, one, and five without edges: every node is alone
EDGELESS = [0, 1, 5]
BOTH = [False, True]


@pytest.mark.parametrize("n", EDGELESS)
@pytest.mark.parametrize("directed", BOTH)
@pytest.mark.parametrize("weighted", BOTH)
def test_edgeless_graphs(n, directed, weighted):
    with load(n, [], directed, weighted) as g:
        assert (g.num_nodes, g.num_edges) == (n, 0)
        zeros, ones = [0] * n, [1] * n

        for kind in ("weak", "strong"):
            c = g.components(kind=kind)
            assert c["count"] == n
            assert len(set(c["components"].tolist())) == n
            eq(c["sizes"], ones)

        if directed:
            for degree in ("in", "out", "total"):
                eq(g.k_core(degree=degree), zeros)
            for kind in ("cycle", "feed_forward"):
                eq(g.k_truss(kind), [])
        else:
            for algorithm in ("liu", "bz"):
                eq(g.k_core(algorithm=algorithm), zeros)
            for algorithm in ("pkt", "burkhardt"):
                eq(g.k_truss(algorithm=algorithm), [])
            eq(g.triangles(), [])
            b = g.biconnectivity()
            assert len(b["heads"]) == 0
            eq(b["articulation_points"], [])
            eq(b["bridges"], [])

        cl = g.clustering()
        eq(cl["local"], zeros)
        # 0 without nodes too, undirected and directed alike
        assert cl["average"] == 0.0
        if not directed:
            assert cl["transitivity"] == 0.0

        # communities: every node its own
        for r in (
            g.louvain(),
            g.leiden(),
            g.label_propagation(),
            g.label_propagation(engine="asynchronous"),
            g.label_propagation(exploration="auto"),
        ):
            assert r["count"] == n
            assert len(set(r["communities"].tolist())) == n
        assert g.modularity(zeros) == 0.0

        # centralities
        for engine in ("exact", "kadabra"):
            eq(g.betweenness(engine=engine)["betweenness"], zeros)
        eq(g.hyperball()["closeness"], zeros)
        eq(g.closeness(), zeros)
        eq(g.harmonic(), zeros)
        eq(g.lin(), ones)
        for engine in ("auto", "partition"):
            ranks = g.pagerank(engine=engine)
            assert len(ranks) == n and np.allclose(ranks, 1 / n if n else 0)
        assert math.isnan(g.degree_assortativity())
        assert math.isnan(g.degree_assortativity(method="spearman"))

        # traversals and local communities, from node 0
        if n == 0:
            for call in (
                lambda: g.bfs(0),
                lambda: g.dijkstra(0),
                lambda: g.dijkstra(0, engine="delta"),
            ):
                with pytest.raises(ValueError, match="out of range"):
                    call()
            with pytest.raises(RuntimeError, match="no nodes"):
                g.spectral_clustering(1)
            return
        unreached = [-1] * (n - 1)
        eq(g.bfs(0), [0, *unreached])
        for engine in ("sequential", "delta"):
            eq(g.dijkstra(0, engine=engine), [0.0] + [math.inf] * (n - 1))
        path = g.shortest_path(0, n - 1)
        if n == 1:
            assert path["distance"] == 0.0
            eq(path["nodes"], [0])
        else:
            assert path is None
        if not directed:
            assert g.hk_relax([0])["size"] == 0
            # the walks can't leave the seed: no community, as HK-Relax's
            assert g.hkpr(0)["size"] == 0


@pytest.mark.parametrize("weighted", BOTH)
def test_a_single_edge(weighted):
    w = 2.5 if weighted else 1.0
    with load(2, [(0, 1)], False, weighted) as g:
        assert g.components()["count"] == 1
        eq(g.k_core(), [1, 1])
        eq(g.k_truss(), [2, 2])
        eq(g.triangles(), [0, 0])
        cl = g.clustering()
        eq(cl["local"], [0.0, 0.0])
        assert (cl["average"], cl["transitivity"]) == (0.0, 0.0)
        b = g.biconnectivity()
        assert len(b["heads"]) == 1
        eq(b["articulation_points"], [])
        assert len(b["bridges"]) == 1
        eq(g.betweenness(engine="exact")["betweenness"], [0.0, 0.0])
        # distances are the weights, on weighted graphs
        eq(g.closeness(), [1 / w, 1 / w])
        eq(g.harmonic(), [1 / w, 1 / w])
        assert np.allclose(g.pagerank(), [0.5, 0.5])
        eq(g.bfs(0), [0, 1])
        eq(g.dijkstra(1), [w, 0.0])
        path = g.shortest_path(0, 1)
        assert path["distance"] == w
        eq(path["nodes"], [0, 1])
        assert g.louvain()["count"] == 1


def test_a_single_arc():
    with load(2, [(0, 1)], True) as g:
        assert g.components()["count"] == 1
        assert g.components(kind="strong")["count"] == 2
        # peeling node 1 (no out-arcs) leaves node 0 none either, and the other way round
        eq(g.k_core(degree="out"), [0, 0])
        eq(g.k_core(degree="in"), [0, 0])
        eq(g.k_core(degree="total"), [1, 1])
        eq(g.bfs(0), [0, 1])
        eq(g.bfs(1), [-1, 0])
        eq(g.bfs(1, direction="in"), [1, 0])
        assert g.shortest_path(1, 0) is None
        eq(g.betweenness(engine="exact")["betweenness"], [0.0, 0.0])
        eq(g.k_truss("cycle"), [2])
        eq(g.k_truss("feed_forward"), [2])
