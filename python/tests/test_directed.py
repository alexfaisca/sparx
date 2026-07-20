"""Directed multigraphs (parallel arcs, self-loops, isolated nodes) against networkx's MultiDiGraph,
tiny graphs, and the cache left behind by views and errors."""

import gc
import os
import random
import shutil

import networkx as nx
import numpy as np
import pytest

import sparx

@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path / ".cache"
    gc.collect()
    left = sorted(os.listdir(tmp_path / ".cache")) if (tmp_path / ".cache").exists() else []
    assert left == [], f"cache files left behind: {left}"

def write(path, n, edges, kind="general"):
    with open(path, "w") as f:
        f.write(f"%%MatrixMarket matrix coordinate pattern {kind}\n{n} {n} {len(edges)}\n")
        for u, v in edges: f.write(f"{u+1} {v+1}\n")
    return str(path)

def multi(seed):
    rnd = random.Random(seed); n = rnd.randint(1, 20)
    arcs = [(rnd.randrange(n), rnd.randrange(n)) for _ in range(rnd.randint(0, 3*n))]
    arcs += arcs[: len(arcs)//3]  # parallel arcs
    arcs += [(u, u) for u in range(n) if rnd.random() < .2]
    M = nx.MultiDiGraph(); M.add_nodes_from(range(n)); M.add_edges_from(arcs)
    return n, arcs, M

SEEDS = range(25)

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("index", ["u32", "u64"])
def test_multigraph_degrees_bfs_components(seed, index):
    n, arcs, M = multi(seed)
    D = nx.DiGraph(M)
    with sparx.Graph(write("m.mtx", n, arcs), index=index) as g:
        assert g.num_edges == len(arcs)
        assert [g.in_degree(u) for u in range(n)] == [M.in_degree(u) for u in range(n)]
        assert [g.degree(u) for u in range(n)] == [M.out_degree(u) for u in range(n)]
        for d, H in {"out": D, "in": D.reverse(), "both": D.to_undirected()}.items():
            for s in range(n):
                L = nx.single_source_shortest_path_length(H, s)
                assert g.bfs(s, direction=d).tolist() == [L.get(u, -1) for u in range(n)], (d, s)
        w = g.components(kind="weak"); st = g.components(kind="strong")
        assert w["count"] == nx.number_weakly_connected_components(D)
        assert st["count"] == nx.number_strongly_connected_components(D)
        assert sorted(st["sizes"].tolist()) == sorted(len(c) for c in nx.strongly_connected_components(D))
        r = g.directed_louvain()
        lab = r["communities"]
        gr = [set(np.flatnonzero(lab == c).tolist()) for c in range(r["count"])]
        q = nx.community.modularity(M, gr) if arcs else 0.0
        assert r["modularity"] == pytest.approx(q, abs=1e-12)
        assert lab.dtype == np.int64

@pytest.mark.parametrize("seed", SEEDS)
def test_multigraph_cores_centralities_clustering(seed):
    n, arcs, M = multi(seed)
    D = nx.DiGraph(M)
    with sparx.Graph(write("m.mtx", n, arcs)) as g:
        # networkx core_number refuses self-loops; compare on graphs without
        if nx.number_of_selfloops(M) == 0:
            core = nx.core_number(M)
            assert g.k_core(direction="both").tolist() == [core[u] for u in range(n)]
        for d, H in {"in": D, "out": D.reverse(), "both": D.to_undirected()}.items():
            if n > 1:
                c = nx.closeness_centrality(H, wf_improved=False)
                np.testing.assert_allclose(g.closeness(direction=d, normalized=True), [c[u] for u in range(n)], atol=1e-12)
                h = nx.harmonic_centrality(H)
                np.testing.assert_allclose(g.harmonic(direction=d, normalized=False), [h[u]/(n-1) for u in range(n)], atol=1e-12)
            g.lin(direction=d); g.hyperball(direction=d, precision=4, normalized=False, gpu=False)
        if nx.number_of_selfloops(M) == 0 and M.number_of_edges() == D.number_of_edges():
            c = nx.clustering(D)
            np.testing.assert_allclose(g.directed_clustering()["local"], [c[u] for u in range(n)], atol=1e-12)

def test_single_node_and_empty():
    for n, arcs in [(1, [(0, 0)]), (3, [(0, 0), (1, 1), (2, 2)]), (2, [(0, 0), (0, 0), (1, 1)])]:
        for idx in ("u32", "u64"):
            with sparx.Graph(write("e.mtx", n, arcs), index=idx) as g:
                r = g.directed_louvain()
                assert r["count"] == n and r["communities"].tolist() == list(range(n)), (n, arcs, r)
                assert r["modularity"] == 0.0 or arcs
                g.components(kind="strong"); g.components()
                g.k_core(direction="in"); g.k_core(direction="out"); g.k_core(direction="both")
                g.directed_clustering(); g.bfs(0, direction="in")
                g.closeness(direction="both"); g.harmonic(direction="in"); g.lin(direction="in")
                g.hyperball(direction="in")

def test_cache_empty_after_views_and_errors():
    g = sparx.Graph(write("d.mtx", 4, [(0, 1), (1, 2), (2, 3)]))
    g.in_degree(0); g.bfs(0, direction="both")
    with pytest.raises(RuntimeError): g.triangles()
    with pytest.raises(RuntimeError): g.louvain()
    with pytest.raises(ValueError): g.k_core(direction="up")
    with pytest.raises(ValueError): g.components(kind="x")
    with pytest.raises(OverflowError): g.in_degree(-1)
    with pytest.raises((ValueError, IndexError)): g.in_degree(4)
    g.close(); g.close()
    with pytest.raises(RuntimeError): g.is_directed
    with pytest.raises(RuntimeError): g.directed_louvain()

def test_keep_cache_with_views(tmp_path):
    g = sparx.Graph(write("d.mtx", 4, [(0, 1), (1, 2), (2, 3)]), keep_cache=True)
    g.k_core(direction="both"); g.in_degree(1)
    cf = g.cache_file
    g.close()
    files = os.listdir(tmp_path / ".cache")
    h = sparx.Graph.open(cf)
    assert h.is_directed and h.in_degree(1) == 1
    assert h.bfs(3, direction="in").tolist() == [3, 2, 1, 0]
    assert h.k_core(direction="both").tolist() == [1, 1, 1, 1]
    h.close()
    shutil.rmtree(tmp_path / ".cache")

def test_k_core_bad_algorithm_with_direction():
    with sparx.Graph(write("d.mtx", 3, [(0, 1)])) as g:
        with pytest.raises(ValueError):
            g.k_core(algorithm="nonsense", direction="out")
