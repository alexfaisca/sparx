"""Checks the bindings against networkx (and brute force references), and their cache handling."""

import gc
import itertools
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import sparx

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ggcat"
ROLES = ("cycle", "out", "in", "middleman")


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    """Runs each test in its own directory (the cache is `./.cache/`), and checks it's left empty."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path / ".cache"
    gc.collect()
    left = sorted(os.listdir(tmp_path / ".cache")) if (tmp_path / ".cache").exists() else []
    assert left == [], f"cache files left behind: {left}"


def write_mtx(path, n, edges, symmetric):
    with open(path, "w") as f:
        kind = "symmetric" if symmetric else "general"
        f.write(f"%%MatrixMarket matrix coordinate pattern {kind}\n{n} {n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u + 1} {v + 1}\n")
    return str(path)


def undirected(G, name="g.mtx"):
    # symmetric files hold each edge once, from the larger node
    return write_mtx(name, G.number_of_nodes(), [(max(e), min(e)) for e in G.edges()], True)


def directed(D, name="d.mtx"):
    return write_mtx(name, D.number_of_nodes(), list(D.edges()), False)


def edge_list(g):
    offsets, targets = g.csr()
    return np.repeat(np.arange(g.num_nodes), np.diff(offsets)), targets


GRAPHS = {
    "karate": nx.karate_club_graph(),
    "gnp": nx.gnp_random_graph(60, 0.12, seed=1),
    # disconnected, with isolated nodes
    "pieces": nx.disjoint_union_all(
        [nx.complete_graph(6), nx.cycle_graph(7), nx.empty_graph(3), nx.petersen_graph()]
    ),
}


@pytest.fixture(params=list(GRAPHS))
def nx_graph(request):
    return nx.convert_node_labels_to_integers(GRAPHS[request.param])


@pytest.fixture(params=["u32", "u64"])
def index(request):
    return request.param


@pytest.fixture
def digraph():
    # with reciprocal edges, which close several triangles
    return nx.gnp_random_graph(35, 0.15, seed=7, directed=True)


def test_csr(nx_graph, index):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        assert (g.num_nodes, g.num_edges, len(g)) == (
            nx_graph.number_of_nodes(),
            2 * nx_graph.number_of_edges(),
            nx_graph.number_of_nodes(),
        )
        assert g.index == index
        src, dst = edge_list(g)
        assert src.dtype == dst.dtype == np.int64
        assert sorted(zip(src.tolist(), dst.tolist())) == sorted(
            itertools.chain(nx_graph.edges(), ((v, u) for u, v in nx_graph.edges()))
        )
        for u in nx_graph:
            assert g.degree(u) == nx_graph.degree(u)
            assert sorted(g.neighbours(u).tolist()) == sorted(nx_graph[u])


@pytest.mark.parametrize("algorithm", ["bz", "liu"])
def test_k_core(nx_graph, index, algorithm):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        core = nx.core_number(nx_graph)
        assert g.k_core(algorithm=algorithm).tolist() == [core[u] for u in range(len(g))]


@pytest.mark.parametrize("width", ["u8", "u16", "u32", "u64"])
def test_triangles(nx_graph, index, width):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        t = g.triangles(width=width)
        assert t.dtype == np.dtype(width.replace("u", "uint"))
        expected = [len(set(nx_graph[u]) & set(nx_graph[v])) for u, v in zip(*edge_list(g))]
        assert t.tolist() == expected


def reference_trussness(G, src, dst):
    trussness, k = {}, 2
    while True:
        truss = nx.k_truss(G, k)
        if truss.number_of_edges() == 0:
            break
        trussness.update({frozenset(e): k for e in truss.edges()})
        k += 1
    return [trussness[frozenset((u, v))] for u, v in zip(src, dst)]


@pytest.mark.parametrize("algorithm", ["pkt", "burkhardt"])
@pytest.mark.parametrize("width", ["u8", "u32"])
def test_k_truss(nx_graph, index, algorithm, width):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        kt = g.k_truss(algorithm=algorithm, width=width)
        assert kt.dtype == np.dtype(width.replace("u", "uint"))
        assert kt.tolist() == reference_trussness(nx_graph, *edge_list(g))


def role_counts(A, u, v):
    """Triangles closed by each `w` with the edge `u -> v`, in each role."""
    counts = {
        "cycle": A[v] * A[:, u],
        "out": A[u] * A[v],
        "in": A[:, u] * A[:, v],
        "middleman": A[u] * A[:, v],
    }
    for c in counts.values():
        c[[u, v]] = 0
    return counts


def selected_roles(kind):
    """The roles of a triangle type's name, e.g. 'cyclefeedforward'."""
    name = kind.replace("feedforward", "outinmiddleman")
    return [r for r in ROLES if r in name]


def test_directed_triangle_types():
    assert len(sparx.DIRECTED_TRIANGLE_TYPES) == 15
    # every non-empty combination of roles, once
    combos = {frozenset(selected_roles(k)) for k in sparx.DIRECTED_TRIANGLE_TYPES}
    assert len(combos) == 15


@pytest.mark.parametrize("kind", sparx.DIRECTED_TRIANGLE_TYPES)
def test_directed_triangles(digraph, index, kind):
    A = nx.to_numpy_array(digraph, nodelist=range(len(digraph)), weight=None)
    with sparx.Graph(directed(digraph), index=index) as g:
        src, dst = edge_list(g)
        expected = [
            sum(role_counts(A, u, v)[r].sum() for r in selected_roles(kind)) for u, v in zip(src, dst)
        ]
        assert g.directed_triangles(kind).tolist() == expected
        assert g.directed_triangles(kind, width="u64").tolist() == expected


def directed_trussness(src, dst, roles):
    """Brute force: the k-truss keeps the edges in `k - 2` of its triangles in a selected role."""
    edge = {(u, v): e for e, (u, v) in enumerate(zip(src, dst))}
    assert len(edge) == len(src), "a simple digraph"
    # each edge's triangles, as pairs of the other two edges
    pairs = {
        "cycle": lambda u, v, w: ((v, w), (w, u)),
        "out": lambda u, v, w: ((u, w), (v, w)),
        "in": lambda u, v, w: ((w, u), (w, v)),
        "middleman": lambda u, v, w: ((u, w), (w, v)),
    }
    nodes = set(src) | set(dst)
    triangles = [
        [
            (edge[a], edge[b])
            for w in nodes - {u, v}
            for r in roles
            for a, b in [pairs[r](u, v, w)]
            if a in edge and b in edge
        ]
        for u, v in zip(src, dst)
    ]
    trussness = [2] * len(src)
    alive, k = set(range(len(src))), 3
    while alive:
        while True:
            peeled = {e for e in alive if sum(a in alive and b in alive for a, b in triangles[e]) < k - 2}
            if not peeled:
                break
            alive -= peeled
        for e in alive:
            trussness[e] = k
        k += 1
    return trussness


@pytest.mark.parametrize("algorithm", ["pkt", "burkhardt"])
@pytest.mark.parametrize("kind", ["cycle", "out", "middleman", "feedforward", "cyclefeedforward"])
def test_directed_k_truss(digraph, algorithm, kind):
    with sparx.Graph(directed(digraph)) as g:
        src, dst = edge_list(g)
        expected = directed_trussness(src.tolist(), dst.tolist(), selected_roles(kind))
        assert g.k_truss(algorithm=algorithm, kind=kind).tolist() == expected


def test_triangle_type_names_are_flexible(digraph):
    with sparx.Graph(directed(digraph)) as g:
        expected = g.directed_triangles("cyclefeedforward").tolist()
        assert g.directed_triangles("Cycle_Feed-Forward").tolist() == expected


def test_clustering(nx_graph, index):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        cc = g.clustering()
        local = nx.clustering(nx_graph)
        assert np.allclose(cc["local"], [local[u] for u in range(len(g))])
        assert cc["average"] == pytest.approx(nx.average_clustering(nx_graph))
        assert cc["transitivity"] == pytest.approx(nx.transitivity(nx_graph))


def test_louvain(nx_graph, index):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        lv = g.louvain()
        communities = lv["communities"]
        assert communities.dtype == np.int64 and len(communities) == len(g)
        parts = [set(np.flatnonzero(communities == c).tolist()) for c in np.unique(communities)]
        assert lv["count"] == len(parts)
        assert lv["modularity"] == pytest.approx(
            nx.community.modularity(nx_graph, parts, weight=None), abs=1e-9
        )
        # Louvain gets close to the best partition networkx's own finds
        best = nx.community.modularity(
            nx_graph, nx.community.louvain_communities(nx_graph, weight=None, seed=0), weight=None
        )
        assert lv["modularity"] >= best - 0.05


def exact_references(G):
    """Per node: sum of distances to, number of (itself included), and sum of inverse distances
    to reachable nodes."""
    n = len(G)
    distances = [nx.single_source_shortest_path_length(G, u) for u in range(n)]
    sums = np.array([sum(d.values()) for d in distances], dtype=float)
    reach = np.array([len(d) for d in distances], dtype=float)
    harmonic = nx.harmonic_centrality(G)
    return sums, reach, np.array([harmonic[u] for u in range(n)])


def ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


CENTRALITY_GRAPHS = {
    "small_world": nx.connected_watts_strogatz_graph(60, 6, 0.2, seed=1),
    "pieces": GRAPHS["pieces"],
}


@pytest.fixture(params=list(CENTRALITY_GRAPHS))
def centrality_graph(request):
    return nx.convert_node_labels_to_integers(CENTRALITY_GRAPHS[request.param])


def expected_centralities(G):
    n = len(G)
    sums, reach, harmonic = exact_references(G)
    return {
        # normalized: over each node's reachable set
        ("closeness", True): ratio(reach - 1, sums),
        ("harmonic", True): ratio(harmonic, reach - 1),
        # not normalized: over the whole graph
        ("closeness", False): ratio(np.full(n, n - 1.0), sums),
        ("harmonic", False): harmonic / (n - 1),
        "lin": np.where(sums > 0, ratio(reach**2, sums), 1.0),
        "distances": sums,
        "harmonic_distances": harmonic,
    }


def test_exact_centralities(centrality_graph):
    G = centrality_graph
    expected = expected_centralities(G)
    with sparx.Graph(undirected(G)) as g:
        for normalized in [True, False]:
            assert np.allclose(g.closeness(normalized=normalized), expected["closeness", normalized])
            assert np.allclose(g.harmonic(normalized=normalized), expected["harmonic", normalized])
        assert np.allclose(g.lin(), expected["lin"])
        # networkx's closeness scales the normalized one by the reachable share of the graph
        _, reach, _ = exact_references(G)
        closeness = nx.closeness_centrality(G)
        assert np.allclose(
            g.closeness() * (reach - 1) / (len(G) - 1), [closeness[u] for u in range(len(G))]
        )


@pytest.mark.parametrize("normalized", [True, False])
def test_hyperball_approximates_exact_centralities(centrality_graph, normalized):
    expected = expected_centralities(centrality_graph)
    with sparx.Graph(undirected(centrality_graph)) as g:
        hb = g.hyperball(precision=12, normalized=normalized)
        for key in ["closeness", "harmonic"]:
            assert np.allclose(hb[key], expected[key, normalized], rtol=0.05), key
        for key in ["lin", "distances", "harmonic_distances"]:
            assert np.allclose(hb[key], expected[key], rtol=0.05), key


@pytest.mark.parametrize("precision", [4, 8, 16])
def test_hyperball_precisions(precision):
    with sparx.Graph(undirected(nx.karate_club_graph())) as g:
        hb = g.hyperball(precision=precision)
        assert set(hb) == {"distances", "harmonic_distances", "closeness", "harmonic", "lin"}
        assert all(len(v) == len(g) and np.isfinite(v).all() for v in hb.values())


def test_bfs(nx_graph, index):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        for source in [0, len(g) - 1]:
            lengths = nx.single_source_shortest_path_length(nx_graph, source)
            assert g.bfs(source).tolist() == [lengths.get(u, -1) for u in range(len(g))]


def test_counts_that_dont_fit_fail_rather_than_wrap():
    # one triangle, with 300 parallel copies of one of its edges
    path = write_mtx("multi.mtx", 3, [(1, 0), (2, 1), (2, 0)] + [(1, 0)] * 300, True)
    with sparx.Graph(path) as g:
        with pytest.raises(RuntimeError, match="more than can be counted"):
            g.triangles()
        assert sorted(set(g.triangles(width="u16").tolist())) == [1, 301]


def test_ggcat_fixture(index):
    with sparx.Graph(FIXTURES / "random_graph_1_5.lz4", index=index) as g:
        assert g.num_nodes > 0 and g.num_edges > 0
        assert g.k_core(algorithm="bz").tolist() == g.k_core(algorithm="liu").tolist()
        assert g.k_truss(algorithm="pkt").tolist() == g.k_truss(algorithm="burkhardt").tolist()


def test_closing_drops_the_cache(cache_dir):
    g = sparx.Graph(undirected(nx.karate_club_graph()))
    g.triangles()
    assert any(cache_dir.iterdir())
    g.close()
    assert not any(cache_dir.iterdir())
    g.close()  # closing twice is fine
    with pytest.raises(RuntimeError, match="closed"):
        g.triangles()
    assert "closed" in repr(g)


def test_garbage_collection_drops_the_cache(cache_dir):
    g = sparx.Graph(undirected(nx.karate_club_graph()))
    g.k_core()
    del g
    gc.collect()
    assert not any(cache_dir.iterdir())


def test_kept_cache_reopens(cache_dir):
    path = undirected(nx.karate_club_graph())
    with sparx.Graph(path, keep_cache=True, index="u32") as g:
        cache_file, expected = g.cache_file, g.k_truss().tolist()
    assert Path(cache_file).exists()
    with sparx.Graph.open(cache_file, index="u32") as g:
        assert g.k_truss().tolist() == expected
    # an opened graph keeps its cache too
    assert Path(cache_file).exists()
    for f in cache_dir.iterdir():
        f.unlink()


def test_graphs_of_the_same_file_are_independent():
    path = undirected(nx.karate_club_graph())
    g1, g2 = sparx.Graph(path), sparx.Graph(path)
    assert g1.cache_id != g2.cache_id
    expected = g1.triangles().tolist()
    g1.close()
    assert g2.triangles().tolist() == expected
    g2.close()


def test_concurrent_calls_on_one_graph():
    with sparx.Graph(undirected(GRAPHS["gnp"])) as g:
        expected = (g.triangles().tolist(), g.k_truss().tolist(), g.k_core().tolist())
        calls = [lambda: g.triangles().tolist(), lambda: g.k_truss().tolist(), lambda: g.k_core().tolist()]
        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(lambda i: calls[i % 3](), range(24)))
        assert all(r == expected[i % 3] for i, r in enumerate(results))


def test_invalid_arguments():
    with sparx.Graph(undirected(nx.karate_club_graph())) as g:
        for call in [
            lambda: g.triangles(width="u128"),
            lambda: g.directed_triangles("triangle"),
            lambda: g.k_truss(algorithm="other"),
            lambda: g.k_truss(kind="loop"),
            lambda: g.k_core(algorithm="other"),
            lambda: g.hyperball(precision=3),
            lambda: g.hyperball(precision=17),
            lambda: g.bfs(len(g)),
            lambda: g.degree(len(g)),
            lambda: g.neighbours(len(g)),
        ]:
            with pytest.raises(ValueError):
                call()
    with pytest.raises(RuntimeError):
        sparx.Graph("missing.mtx")
    with pytest.raises(RuntimeError):
        sparx.Graph(undirected(nx.karate_club_graph()), index="u16")


@pytest.mark.skipif(sparx.GPU, reason="built with GPU support")
def test_gpu_needs_the_feature():
    with sparx.Graph(undirected(nx.karate_club_graph())) as g:
        with pytest.raises(ValueError, match="GPU"):
            g.triangles(gpu=True)
        with pytest.raises(ValueError, match="GPU"):
            g.hyperball(gpu=True)


@pytest.mark.skipif(not sparx.GPU, reason="built without GPU support")
def test_gpu_matches_cpu():
    with sparx.Graph(undirected(GRAPHS["gnp"])) as g:
        assert g.triangles(gpu=True, width="u16").tolist() == g.triangles(width="u16").tolist()
        cpu, gpu = g.hyperball(precision=8), g.hyperball(precision=8, gpu=True)
        for key in cpu:
            assert np.allclose(gpu[key], cpu[key], rtol=1e-5), key
