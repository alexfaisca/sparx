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
    with sparx.DiGraph(write("m.mtx", n, arcs), index=index) as g:
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
        r = g.louvain()
        lab = r["communities"]
        gr = [set(np.flatnonzero(lab == c).tolist()) for c in range(r["count"])]
        q = nx.community.modularity(M, gr) if arcs else 0.0
        assert r["modularity"] == pytest.approx(q, abs=1e-12)
        assert lab.dtype == np.int64

@pytest.mark.parametrize("seed", SEEDS)
def test_multigraph_cores_centralities_clustering(seed):
    n, arcs, M = multi(seed)
    D = nx.DiGraph(M)
    with sparx.DiGraph(write("m.mtx", n, arcs)) as g:
        # networkx core_number refuses self-loops; compare on graphs without
        if nx.number_of_selfloops(M) == 0:
            core = nx.core_number(M)
            assert g.k_core(degree="total").tolist() == [core[u] for u in range(n)]
        for d, H in {"in": D, "out": D.reverse(), "both": D.to_undirected()}.items():
            if n > 1:
                c = nx.closeness_centrality(H, wf_improved=False)
                np.testing.assert_allclose(g.closeness(direction=d, normalized=True), [c[u] for u in range(n)], atol=1e-12)
                h = nx.harmonic_centrality(H)
                np.testing.assert_allclose(g.harmonic(direction=d, normalized=False), [h[u]/(n-1) for u in range(n)], atol=1e-12)
            g.lin(direction=d); g.hyperball(direction=d, precision=4, normalized=False, gpu=False)
        if nx.number_of_selfloops(M) == 0 and M.number_of_edges() == D.number_of_edges():
            c = nx.clustering(D)
            np.testing.assert_allclose(g.clustering()["local"], [c[u] for u in range(n)], atol=1e-12)

def test_single_node_and_empty():
    for n, arcs in [(1, [(0, 0)]), (3, [(0, 0), (1, 1), (2, 2)]), (2, [(0, 0), (0, 0), (1, 1)])]:
        for idx in ("u32", "u64"):
            with sparx.DiGraph(write("e.mtx", n, arcs), index=idx) as g:
                r = g.louvain()
                assert r["count"] == n and r["communities"].tolist() == list(range(n)), (n, arcs, r)
                assert r["modularity"] == 0.0 or arcs
                g.components(kind="strong"); g.components()
                g.k_core(degree="in"); g.k_core(degree="out"); g.k_core(degree="total")
                g.clustering(); g.bfs(0, direction="in")
                g.closeness(direction="both"); g.harmonic(direction="in"); g.lin(direction="in")
                g.hyperball(direction="in")

def test_cache_empty_after_views_and_errors():
    g = sparx.DiGraph(write("d.mtx", 4, [(0, 1), (1, 2), (2, 3)]))
    g.in_degree(0); g.bfs(0, direction="both")
    with pytest.raises(AttributeError): g.triangles()
    with pytest.raises(RuntimeError, match="DiGraph"): sparx.Graph(write("d2.mtx", 4, [(0, 1)]))
    with pytest.raises(ValueError): g.k_core(degree="up")
    with pytest.raises(ValueError): g.k_truss("up")
    with pytest.raises(ValueError): g.components(kind="x")
    with pytest.raises(OverflowError): g.in_degree(-1)
    with pytest.raises((ValueError, IndexError)): g.in_degree(4)
    g.close(); g.close()
    with pytest.raises(RuntimeError): g.is_symmetric
    with pytest.raises(RuntimeError): g.louvain()

def test_keep_cache_with_views(tmp_path):
    g = sparx.DiGraph(write("d.mtx", 4, [(0, 1), (1, 2), (2, 3)]), keep_cache=True)
    g.k_core(degree="total"); g.in_degree(1)
    cf = g.cache_file
    g.close()
    files = os.listdir(tmp_path / ".cache")
    h = sparx.DiGraph.open(cf)
    assert h.is_directed and not h.is_symmetric and h.in_degree(1) == 1
    assert h.bfs(3, direction="in").tolist() == [3, 2, 1, 0]
    assert h.k_core(degree="total").tolist() == [1, 1, 1, 1]
    h.close()
    shutil.rmtree(tmp_path / ".cache")

def test_k_core_bad_degree():
    with sparx.DiGraph(write("d.mtx", 3, [(0, 1)])) as g:
        with pytest.raises(ValueError):
            g.k_core(degree="nonsense")


# --- conversions between Graph and DiGraph ------------------------------------------------------


def csr_arcs(g):
    offsets, targets = g.csr()
    src = np.repeat(np.arange(g.num_nodes), np.diff(offsets))
    return sorted(zip(src.tolist(), targets.tolist()))


def undirected_multigraph(n, arcs):
    """networkx's `MultiDiGraph.to_undirected()`: every arc an edge, reciprocal and parallel arcs
    parallel edges (`mode="each"`)."""
    U = nx.MultiGraph(); U.add_nodes_from(range(n)); U.add_edges_from(arcs)
    return U


def undirected_graphs(n, arcs):
    """The undirected graphs of each mode, by networkx."""
    D = nx.DiGraph(); D.add_nodes_from(range(n)); D.add_edges_from(arcs)
    return {
        "collapse": D.to_undirected(),
        "each": undirected_multigraph(n, arcs),
        # networkx keeps self-loops too: a self-loop is its own reverse
        "mutual": D.to_undirected(reciprocal=True),
    }


def multigraph_arcs(U):
    """A (Multi)Graph's edges as sparx stores them: an arc each way, self-loops once."""
    out = []
    for u, v in U.edges():
        out += [(u, v), (v, u)] if u != v else [(u, u)]
    return sorted(out)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("index", ["u32", "u64"])
def test_to_undirected_matches_networkx(seed, index):
    n, arcs, M = multi(seed)
    with sparx.DiGraph(write("m.mtx", n, arcs), index=index) as d:
        assert d.to_undirected().csr()[1].tolist() == d.to_undirected(mode="collapse").csr()[1].tolist()
        for mode, U in undirected_graphs(n, arcs).items():
            u = d.to_undirected(mode=mode)
            assert isinstance(u, sparx.Graph) and u.index == index and u.threads == d.threads
            assert not u.is_directed and u.is_symmetric
            assert csr_arcs(u) == multigraph_arcs(U), mode
            assert u.num_nodes == n and u.num_edges == len(multigraph_arcs(U))
            loops = {w: sum(1 for a, b in U.edges() if a == b == w) for w in range(n)}
            assert [u.degree(w) for w in range(n)] == [U.degree(w) - loops[w] for w in range(n)]
            for s in range(n):
                L = nx.single_source_shortest_path_length(U, s)
                assert u.bfs(s).tolist() == [L.get(w, -1) for w in range(n)], (mode, s)
            cc = u.components()
            assert cc["count"] == nx.number_connected_components(U)
            if nx.number_of_selfloops(U) == 0:
                core = nx.core_number(U)
                assert u.k_core().tolist() == [core[w] for w in range(n)], mode
            if n > 1:
                c = nx.closeness_centrality(U, wf_improved=False)
                np.testing.assert_allclose(u.closeness(), [c[w] for w in range(n)], atol=1e-12)
            if mode == "collapse":
                # what direction="both" runs on
                for s in range(n):
                    assert u.bfs(s).tolist() == d.bfs(s, direction="both").tolist()
                assert cc["components"].tolist() == d.components(kind="weak")["components"].tolist()
                if n > 1:
                    np.testing.assert_allclose(u.closeness(), d.closeness(direction="both"), atol=1e-12)
            if mode == "each" and nx.number_of_selfloops(M) == 0:
                # total degree cores count every arc
                assert u.k_core().tolist() == d.k_core(degree="total").tolist()
            # converting back: a symmetric DiGraph of the same arcs
            back = u.to_directed()
            assert back.is_directed and back.is_symmetric
            assert csr_arcs(back) == csr_arcs(u)
            assert [back.in_degree(w) for w in range(n)] == [u.degree(w) for w in range(n)]
            # closing a view leaves its parent (and its cache) alone
            back.close(); u.close()
            assert csr_arcs(d.to_undirected(mode=mode)) == multigraph_arcs(U)
        with pytest.raises(ValueError, match="unknown undirected mode"):
            d.to_undirected(mode="reciprocal")


GRAPHS = {
    "karate": nx.karate_club_graph(),
    "gnp": nx.gnp_random_graph(40, 0.12, seed=1),
    "pieces": nx.disjoint_union_all([nx.complete_graph(5), nx.cycle_graph(6), nx.empty_graph(2)]),
}


@pytest.mark.parametrize("name", list(GRAPHS))
@pytest.mark.parametrize("index", ["u32", "u64"])
def test_to_directed_matches_networkx(name, index):
    G = nx.convert_node_labels_to_integers(GRAPHS[name])
    D = G.to_directed()
    n = len(G)
    edges = [(max(e), min(e)) for e in G.edges()]
    with sparx.Graph(write("g.mtx", n, edges, kind="symmetric"), index=index) as g:
        d = g.to_directed()
        assert isinstance(d, sparx.DiGraph) and d.index == index
        assert d.is_directed and d.is_symmetric
        assert csr_arcs(d) == csr_arcs(g) == sorted(D.edges())
        assert [d.in_degree(w) for w in range(n)] == [D.in_degree(w) for w in range(n)]
        assert [sorted(d.in_neighbours(w).tolist()) for w in range(n)] == [sorted(D.predecessors(w)) for w in range(n)]
        for direction in ["out", "in", "both"]:
            assert d.bfs(0, direction=direction).tolist() == g.bfs(0).tolist()
        # in- and out-cores of a symmetric DiGraph are the usual cores; total degree cores count
        # both arcs of each edge, as networkx's core_number(D) does
        core = nx.core_number(G)
        for degree in ["in", "out"]:
            assert d.k_core(degree=degree).tolist() == [core[w] for w in range(n)] == g.k_core().tolist()
        directed_core = nx.core_number(D)
        assert d.k_core(degree="total").tolist() == [directed_core[w] for w in range(n)]
        assert d.k_core(degree="total").tolist() == [2 * core[w] for w in range(n)]
        c = nx.clustering(D)
        np.testing.assert_allclose(d.clustering()["local"], [c[w] for w in range(n)], atol=1e-12)
        np.testing.assert_allclose(d.clustering()["local"], g.clustering()["local"], atol=1e-12)
        r = d.louvain()
        groups = [set(np.flatnonzero(r["communities"] == k).tolist()) for k in range(r["count"])]
        assert r["modularity"] == pytest.approx(nx.community.modularity(D, groups, weight=None), abs=1e-12)
        assert d.components(kind="strong")["count"] == nx.number_connected_components(G)
        # its collapsed and mutual undirected graphs are the same entry
        for mode in ["collapse", "mutual"]:
            u = d.to_undirected(mode=mode)
            assert u.cache_file == g.cache_file == d.cache_file
            assert csr_arcs(u) == csr_arcs(g)
            assert u.k_truss().tolist() == g.k_truss().tolist()
        # while each arc is an edge of its "each" one: every edge twice
        e = d.to_undirected(mode="each")
        assert e.cache_file != g.cache_file
        assert csr_arcs(e) == sorted(csr_arcs(g) * 2)


def test_views_need_their_parent_open(cache_dir):
    path = write("d.mtx", 4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    d = sparx.DiGraph(path)
    u = d.to_undirected()
    expected = u.triangles().tolist()
    assert any(cache_dir.iterdir())
    d.close()
    # the parent dropped its cache (the undirected graph included)
    assert not any(cache_dir.iterdir())
    for call in [u.triangles, u.k_core, lambda: u.num_nodes, u.csr, lambda: u.bfs(0), u.to_directed]:
        with pytest.raises(RuntimeError, match="converted from is closed"):
            call()
    assert "closed" in repr(u)
    u.close(); u.close()
    with pytest.raises(RuntimeError, match="graph is closed"):
        u.triangles()

    # views of views, and parents garbage collected first
    g = sparx.Graph(write("s.mtx", 3, [(1, 0), (2, 1), (2, 0)], kind="symmetric"))
    dd = g.to_directed()
    uu = dd.to_undirected()
    assert uu.triangles().tolist() == [1] * 6
    dd.close()
    with pytest.raises(RuntimeError, match="converted from is closed"):
        uu.triangles()
    del g
    gc.collect()
    assert not any(cache_dir.iterdir())
    with pytest.raises(RuntimeError):
        sparx.DiGraph(path).to_undirected().triangles()
    assert expected == [1] * 6 + [0] * 2


def test_views_and_parents_run_concurrently():
    from concurrent.futures import ThreadPoolExecutor

    rnd = random.Random(3)
    arcs = [(rnd.randrange(60), rnd.randrange(60)) for _ in range(400)]
    with sparx.DiGraph(write("d.mtx", 60, arcs)) as d:
        u = d.to_undirected()
        uu = u.to_directed()
        expected = (d.bfs(0, direction="both").tolist(), u.bfs(0).tolist(), uu.bfs(0).tolist())
        assert expected[0] == expected[1] == expected[2]
        calls = [lambda: d.bfs(0, direction="both").tolist(), lambda: u.bfs(0).tolist(), lambda: uu.bfs(0).tolist()]
        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(lambda i: calls[i % 3](), range(30)))
        assert all(r == expected[0] for r in results)


def test_kept_digraph_keeps_its_undirected_graph(tmp_path):
    arcs = [(0, 1), (1, 2), (2, 0), (0, 2)]
    with sparx.DiGraph(write("d.mtx", 3, arcs), keep_cache=True) as d:
        cf, expected = d.cache_file, csr_arcs(d.to_undirected())
    with sparx.DiGraph.open(cf) as d:
        assert csr_arcs(d.to_undirected()) == expected == multigraph_arcs(undirected_graphs(3, arcs)["collapse"])
        for mode, U in undirected_graphs(3, arcs).items():
            assert csr_arcs(d.to_undirected(mode=mode)) == multigraph_arcs(U)
    shutil.rmtree(tmp_path / ".cache")


def test_symmetric_to_undirected():
    """A symmetric DiGraph without parallel arcs is its own collapsed and mutual undirected graph;
    its "each" one doubles every edge. Directed or not, the collapsed graph has one edge per linked
    pair."""
    arcs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
    with sparx.DiGraph(write("s.mtx", 3, arcs)) as d:
        assert d.is_symmetric
        for mode in ["collapse", "mutual"]:
            u = d.to_undirected(mode=mode)
            assert u.cache_file == d.cache_file
            assert csr_arcs(u) == sorted(arcs)
            assert u.triangles().tolist() == [1] * 6
        each = d.to_undirected(mode="each")
        assert each.cache_file != d.cache_file
        assert csr_arcs(each) == sorted(arcs * 2)
        # total degree cores count both arcs of each edge, as networkx's
        D = nx.DiGraph(arcs)
        assert d.k_core(degree="total").tolist() == [nx.core_number(D)[w] for w in range(3)] == [4, 4, 4]
    # one more arc: no longer symmetric, but the collapsed and mutual graphs are the same
    with sparx.DiGraph(write("d.mtx", 3, arcs + [(0, 1)])) as d:
        assert not d.is_symmetric
        for mode in ["collapse", "mutual"]:
            u = d.to_undirected(mode=mode)
            assert u.cache_file != d.cache_file
            assert csr_arcs(u) == sorted(arcs)
        assert csr_arcs(d.to_undirected(mode="each")) == multigraph_arcs(undirected_multigraph(3, arcs + [(0, 1)]))
    # a single arc of a pair: collapsed, not mutual
    with sparx.DiGraph(write("p.mtx", 3, [(0, 1), (1, 2), (2, 1), (2, 2)])) as d:
        assert csr_arcs(d.to_undirected()) == [(0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]
        assert csr_arcs(d.to_undirected(mode="mutual")) == [(1, 2), (2, 1), (2, 2)]


def test_real_valued_matrix_market_files_load_their_structure(tmp_path):
    """Real (and complex) entries' values are ignored: the structure is the pattern file's."""
    arcs = [(1, 0), (2, 0), (2, 2), (3, 1)]
    def write(name, field, symmetry, values):
        path = tmp_path / name
        lines = [f"%%MatrixMarket matrix coordinate {field} {symmetry}", f"4 4 {len(arcs)}"]
        lines += [f"{u + 1} {v + 1}{value}" for (u, v), value in zip(arcs, values)]
        path.write_text("\n".join(lines) + "\n")
        return str(path)
    for symmetry, cls in [("general", sparx.DiGraph), ("symmetric", sparx.Graph)]:
        with cls(write(f"p_{symmetry}.mtx", "pattern", symmetry, [""] * 4)) as reference:
            want = [a.tolist() for a in reference.csr()]
        for field, values in [("real", [" -1.5", " 2e-3", " 0", " 7.25"]), ("complex", [" 1 -2", " 0.5 0", " 3 0", " -1e2 4"])]:
            with cls(write(f"{field}_{symmetry}.mtx", field, symmetry, values)) as g:
                assert [a.tolist() for a in g.csr()] == want, (field, symmetry)
    with sparx.Graph(write("h.mtx", "complex", "hermitian", [" 1 1"] * 4)) as g:
        assert g.is_symmetric
