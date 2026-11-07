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
@pytest.mark.parametrize("width", ["u8", "u16", "u32", "u64"])
def test_k_core(nx_graph, index, algorithm, width):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        core = nx.core_number(nx_graph)
        kc = g.k_core(algorithm=algorithm, width=width)
        assert kc.dtype == np.dtype(width.replace("u", "uint"))
        assert kc.tolist() == [core[u] for u in range(len(g))]
        if width == "u8":
            assert g.k_core(algorithm=algorithm).dtype == np.uint8


@pytest.mark.parametrize("algorithm", ["bz", "liu"])
def test_degrees_that_dont_fit_a_coreness_width_fail_rather_than_wrap(algorithm):
    for leaves, narrow, wide in [(255, None, "u8"), (256, "u8", "u16")]:
        star = nx.star_graph(leaves)
        with sparx.Graph(undirected(star)) as g:
            if narrow:
                with pytest.raises(RuntimeError, match="coreness type"):
                    g.k_core(algorithm=algorithm, width=narrow)
            assert g.k_core(algorithm=algorithm, width=wide).tolist() == [1] * (leaves + 1)
    # every node of degree 256, and coreness 256
    clique = nx.complete_graph(257)
    with sparx.Graph(undirected(clique)) as g:
        with pytest.raises(RuntimeError, match="coreness type"):
            g.k_core(algorithm=algorithm)
        assert g.k_core(algorithm=algorithm, width="u16").tolist() == [256] * 257


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
    with sparx.DiGraph(directed(digraph), index=index) as g:
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
    with sparx.DiGraph(directed(digraph)) as g:
        src, dst = edge_list(g)
        expected = directed_trussness(src.tolist(), dst.tolist(), selected_roles(kind))
        assert g.k_truss(kind, algorithm=algorithm).tolist() == expected


def test_triangle_type_names_are_flexible(digraph):
    with sparx.DiGraph(directed(digraph)) as g:
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
            lambda: g.k_truss(algorithm="other"),
            lambda: g.k_core(algorithm="other"),
            lambda: g.k_core(width="u128"),
            lambda: g.hyperball(precision=3),
            lambda: g.hyperball(precision=17),
            lambda: g.bfs(len(g)),
            lambda: g.degree(len(g)),
            lambda: g.neighbours(len(g)),
        ]:
            with pytest.raises(ValueError):
                call()
    with sparx.DiGraph(directed(nx.gnp_random_graph(10, 0.3, seed=0, directed=True))) as d:
        for call in [
            lambda: d.directed_triangles("triangle"),
            lambda: d.directed_triangles("cycle", width="u7"),
            lambda: d.k_truss("loop"),
            lambda: d.k_truss("cycle", algorithm="other"),
            lambda: d.k_core(degree="both"),
            lambda: d.k_core(width="usize"),
            lambda: d.hyperball(precision=3),
            lambda: d.hyperball(direction="up"),
            lambda: d.closeness(direction="up"),
            lambda: d.bfs(len(d)),
            lambda: d.bfs(0, direction="up"),
            lambda: d.in_degree(len(d)),
            lambda: d.in_neighbours(len(d)),
            lambda: d.components(kind="semi"),
        ]:
            with pytest.raises(ValueError):
                call()
    for cls in (sparx.Graph, sparx.DiGraph):
        with pytest.raises(RuntimeError):
            cls("missing.mtx")


@pytest.mark.parametrize("cls", [sparx.Graph, sparx.DiGraph])
def test_bad_index_is_a_value_error(cls):
    path = undirected(nx.karate_club_graph())
    for index in ["u16", "U32", "", "usize"]:
        with pytest.raises(ValueError, match="unknown index type"):
            cls(path, index=index)
        with pytest.raises(ValueError, match="unknown index type"):
            cls(path, index=index, keep_cache=True)
        # checked before looking for the file
        with pytest.raises(ValueError, match="unknown index type"):
            cls.open("missing.mmap", index=index)
    with cls(path, keep_cache=True) as g:
        cache_file = g.cache_file
    with pytest.raises(ValueError, match="unknown index type"):
        cls.open(cache_file, index="u16")
    cls.open(cache_file).close()
    for f in Path(cache_file).parent.iterdir():
        f.unlink()


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


# --- directed graphs ---------------------------------------------------------------------------


def directed(G, name="d.mtx"):
    return write_mtx(name, G.number_of_nodes(), list(G.edges()), False)


def random_digraphs():
    for seed, (n, p) in enumerate([(12, 0.15), (25, 0.08), (40, 0.05), (30, 0.2)]):
        yield nx.gnp_random_graph(n, p, seed=seed, directed=True)


def by_smallest_node(components, n):
    """Per node component ids, components numbered by their smallest node."""
    labels = [0] * n
    for c, nodes in enumerate(sorted(components, key=min)):
        for u in nodes:
            labels[u] = c
    return labels


def test_symmetry_is_recorded(tmp_path):
    with sparx.DiGraph(write_mtx("a.mtx", 3, [(0, 1), (1, 2)], False)) as g:
        assert g.is_directed and not g.is_symmetric
    for path in [write_mtx("b.mtx", 3, [(0, 1), (1, 0)], False), write_mtx("c.mtx", 3, [(1, 0), (2, 1)], True)]:
        with sparx.DiGraph(path) as g:
            assert g.is_directed and g.is_symmetric
        with sparx.Graph(path) as g:
            assert not g.is_directed and g.is_symmetric


@pytest.mark.parametrize("G", list(random_digraphs()))
def test_directed_degrees_and_bfs(G):
    n = G.number_of_nodes()
    views = {"out": G, "in": G.reverse(), "both": G.to_undirected()}
    with sparx.DiGraph(directed(G)) as g:
        assert [g.in_degree(u) for u in range(n)] == [G.in_degree(u) for u in range(n)]
        for direction, H in views.items():
            for source in range(0, n, 5):
                lengths = nx.single_source_shortest_path_length(H, source)
                want = [lengths.get(u, -1) for u in range(n)]
                assert g.bfs(source, direction=direction).tolist() == want, (direction, source)


@pytest.mark.parametrize("G", list(random_digraphs()))
def test_directed_centralities_against_networkx(G):
    n = G.number_of_nodes()
    # networkx's closeness and harmonic centralities use distances *to* a node: "in"
    views = {"in": G, "out": G.reverse(), "both": G.to_undirected()}
    with sparx.DiGraph(directed(G)) as g:
        for direction, H in views.items():
            closeness = nx.closeness_centrality(H, wf_improved=False)
            got = g.closeness(direction=direction, normalized=True)
            np.testing.assert_allclose(got, [closeness[u] for u in range(n)], atol=1e-12)
            harmonic = nx.harmonic_centrality(H)
            got = g.harmonic(direction=direction, normalized=False)
            np.testing.assert_allclose(got, [harmonic[u] / (n - 1) for u in range(n)], atol=1e-12)
            # HyperBall estimates them
            hb = g.hyperball(direction=direction, precision=12)
            np.testing.assert_allclose(hb["closeness"], g.closeness(direction=direction), rtol=0.1, atol=1e-9)


@pytest.mark.parametrize("G", list(random_digraphs()))
def test_components_against_networkx(G):
    n = G.number_of_nodes()
    with sparx.DiGraph(directed(G)) as g:
        weak = g.components(kind="weak")
        assert weak["components"].tolist() == by_smallest_node(nx.weakly_connected_components(G), n)
        assert weak["count"] == nx.number_weakly_connected_components(G)
        assert weak["sizes"].sum() == n
        strong = g.components(kind="strong")
        assert strong["components"].tolist() == by_smallest_node(nx.strongly_connected_components(G), n)
        assert strong["count"] == nx.number_strongly_connected_components(G)
        with pytest.raises(ValueError):
            g.components(kind="semi")


def test_components_of_undirected_graphs():
    G = nx.gnp_random_graph(40, 0.04, seed=3)
    with sparx.Graph(undirected(G)) as g:
        want = by_smallest_node(nx.connected_components(G), 40)
        assert g.components()["components"].tolist() == want
        assert g.components(kind="strong")["components"].tolist() == want


def brute_force_cores(G, degree):
    """Coreness by repeated peeling, `degree(H, u)` within the remaining subgraph `H`."""
    H, core, k = G.copy(), {}, 0
    while H.number_of_nodes():
        while True:
            low = [u for u in H if degree(H, u) < k]
            if not low:
                break
            H.remove_nodes_from(low)
        for u in H:
            core[u] = k
        k += 1
    return [core[u] for u in range(G.number_of_nodes())]


@pytest.mark.parametrize("G", list(random_digraphs()))
def test_directed_k_cores(G):
    with sparx.DiGraph(directed(G)) as g:
        core = nx.core_number(G)
        assert g.k_core().tolist() == [core[u] for u in range(G.number_of_nodes())]
        assert g.k_core(degree="total").tolist() == [core[u] for u in range(G.number_of_nodes())]
        assert g.k_core(degree="in").tolist() == brute_force_cores(G, lambda H, u: H.in_degree(u))
        assert g.k_core(degree="out").tolist() == brute_force_cores(G, lambda H, u: H.out_degree(u))
        for width in ["u8", "u16", "u32", "u64"]:
            for degree in ["in", "out", "total"]:
                kc = g.k_core(degree=degree, width=width)
                assert kc.dtype == np.dtype(width.replace("u", "uint"))
                assert kc.tolist() == g.k_core(degree=degree).tolist()


def test_directed_degrees_that_dont_fit_a_coreness_width_fail_rather_than_wrap():
    # every node has in- and out-degree 255: total degree 510
    D = nx.complete_graph(256, create_using=nx.DiGraph)
    with sparx.DiGraph(directed(D)) as d:
        assert d.k_core(degree="in").tolist() == [255] * 256
        assert d.k_core(degree="out").tolist() == [255] * 256
        with pytest.raises(RuntimeError, match="coreness type"):
            d.k_core()
        assert d.k_core(width="u16").tolist() == [510] * 256


@pytest.mark.parametrize("G", list(random_digraphs()))
def test_directed_clustering_against_networkx(G):
    n = G.number_of_nodes()
    with sparx.DiGraph(directed(G)) as g:
        c = g.clustering()
        want = nx.clustering(G)
        np.testing.assert_allclose(c["local"], [want[u] for u in range(n)], atol=1e-12)
        assert c["average"] == pytest.approx(nx.average_clustering(G))


def test_graph_refuses_directed_files(cache_dir):
    path = write_mtx("d.mtx", 3, [(0, 1), (1, 2), (0, 2)], False)
    for index in ["u32", "u64"]:
        with pytest.raises(RuntimeError, match="DiGraph") as e:
            sparx.Graph(path, index=index)
        assert "arc 0 -> 1 has no reverse" in str(e.value)
        # the refused entry isn't left behind
        assert not cache_dir.exists() or not any(cache_dir.iterdir())
    # and an existing directed entry can't be opened as one either, nor is it removed for it
    with sparx.DiGraph(path, keep_cache=True) as d:
        cache_file = d.cache_file
    with pytest.raises(RuntimeError, match="DiGraph"):
        sparx.Graph.open(cache_file)
    assert Path(cache_file).exists()
    sparx.DiGraph.open(cache_file).close()
    for f in cache_dir.iterdir():
        f.unlink()


def test_directed_and_undirected_methods_are_split():
    for name in ["triangles", "to_directed"]:
        assert hasattr(sparx.Graph, name) and not hasattr(sparx.DiGraph, name)
    for name in ["in_degree", "in_neighbours", "directed_triangles", "to_undirected"]:
        assert hasattr(sparx.DiGraph, name) and not hasattr(sparx.Graph, name)
    for name in ["directed_clustering", "directed_louvain"]:
        assert not hasattr(sparx.Graph, name) and not hasattr(sparx.DiGraph, name)
    with sparx.DiGraph(write_mtx("d.mtx", 3, [(0, 1), (1, 2), (0, 2)], False)) as g:
        with pytest.raises(ValueError, match="unknown direction"):
            g.bfs(0, direction="sideways")
        with pytest.raises(ValueError, match="unknown core degree"):
            g.k_core(degree="both")
        with pytest.raises(TypeError):
            g.k_truss()  # kind is required
        with pytest.raises(TypeError):
            g.k_core(algorithm="bz")
    with sparx.Graph(undirected(nx.karate_club_graph())) as g:
        with pytest.raises(TypeError):
            g.k_truss(kind="cycle")
        with pytest.raises(TypeError):
            g.k_core(direction="in")
        with pytest.raises(TypeError):
            g.bfs(0, direction="in")


@pytest.mark.parametrize("G", list(random_digraphs()))
def test_directed_louvain_against_networkx_modularity(G):
    n = G.number_of_nodes()
    with sparx.DiGraph(directed(G)) as g:
        result = g.louvain()
        labels = result["communities"].tolist()
        groups = [{u for u in range(n) if labels[u] == c} for c in range(result["count"])]
        assert result["modularity"] == pytest.approx(nx.community.modularity(G, groups), abs=1e-12)
        singletons = [{u} for u in range(n)]
        assert result["modularity"] >= nx.community.modularity(G, singletons) - 1e-12


def test_in_degree_releases_the_gil():
    """The first `in_degree` builds the transpose: other Python threads run meanwhile."""
    import threading
    import time

    n, m = 200_000, 2_000_000
    arcs = np.random.default_rng(0).integers(1, n + 1, size=(m, 2))
    with open("big.mtx", "w") as f:
        f.write(f"%%MatrixMarket matrix coordinate pattern general\n{n} {n} {m}\n")
        f.write("%d %d\n" * m % tuple(arcs.ravel().tolist()))
    with sparx.DiGraph("big.mtx") as d:
        ticks, done = [0], threading.Event()

        def tick():
            while not done.is_set():
                ticks[0] += 1
                time.sleep(0.0005)

        ticker = threading.Thread(target=tick)
        ticker.start()
        time.sleep(0.02)
        before = ticks[0]
        degree = d.in_degree(0)
        during = ticks[0] - before
        done.set()
        ticker.join()
        assert degree == int((arcs[:, 1] == 1).sum())
        # holding the GIL, the ticker wouldn't tick at all while the transpose is built
        assert during >= 10, during
        # and is_symmetric (an O(E) scan, for entries of earlier versions) doesn't hold it either
        assert d.is_symmetric is False


def cache_listing(cache_dir):
    return sorted(str(p.relative_to(cache_dir)) for p in cache_dir.rglob("*"))


def test_failed_calls_leave_kept_caches_as_they_were(cache_dir):
    # one triangle, with 300 parallel copies of one of its edges: more triangles than u8 counts
    g = sparx.Graph(write_mtx("s.mtx", 3, [(1, 0), (2, 1), (2, 0)] + [(1, 0)] * 300, True), keep_cache=True)
    d = sparx.DiGraph(write_mtx("d.mtx", 3, [(0, 1)] * 300 + [(1, 2), (2, 0)], False), keep_cache=True)
    # (its collapsed undirected graph has no parallel edges: its "each" one has 300)
    u, gd = d.to_undirected(mode="each"), g.to_directed()
    failing = {
        "triangles": lambda: g.triangles(),
        "k_truss pkt": lambda: g.k_truss(),
        "k_truss burkhardt": lambda: g.k_truss(algorithm="burkhardt"),
        "clustering": g.clustering,
        "directed triangles": lambda: d.directed_triangles("cycle"),
        "directed k_truss pkt": lambda: d.k_truss("cycle"),
        "directed k_truss burkhardt": lambda: d.k_truss("cycle", algorithm="burkhardt"),
        "converted triangles": lambda: u.triangles(),
        "converted k_truss": lambda: u.k_truss(),
        "converted clustering": u.clustering,
        "converted directed triangles": lambda: gd.directed_triangles("cyclefeedforward"),
        "converted directed k_truss": lambda: gd.k_truss("out", algorithm="burkhardt"),
    }
    for name, call in failing.items():
        before = cache_listing(cache_dir)
        with pytest.raises(RuntimeError, match="more than 255"):
            call()
        assert cache_listing(cache_dir) == before, name
    # (nodes of degree 300 or more too); total degree cores peel a graph's "each" view, which
    # a failure leaves, like a successful call does, until the graph is closed
    gd.to_undirected(mode="each")
    failing_cores = {
        "k_core bz": lambda: g.k_core(),
        "k_core liu": lambda: g.k_core(algorithm="liu"),
        "directed k_core in": lambda: d.k_core(degree="in"),
        "directed k_core out": lambda: d.k_core(degree="out"),
        "directed k_core total": lambda: d.k_core(),
        "converted k_core": lambda: u.k_core(algorithm="liu"),
        "converted directed k_core": lambda: gd.k_core(),
    }
    for name, call in failing_cores.items():
        before = cache_listing(cache_dir)
        with pytest.raises(RuntimeError, match="coreness type"):
            call()
        assert cache_listing(cache_dir) == before, name
    assert g.k_core(width="u16").tolist() == g.k_core(algorithm="liu", width="u16").tolist()
    # the graphs are intact, and wider counts work
    assert sorted(set(g.triangles(width="u16").tolist())) == [1, 301]
    assert sorted(set(d.directed_triangles("cycle", width="u16").tolist())) == [1, 300]
    for h in (gd, u, g, d):
        h.close()
    for f in sorted(cache_dir.rglob("*"), reverse=True):
        f.rmdir() if f.is_dir() else f.unlink()


@pytest.mark.parametrize("alpha", [0.85, 0.5])
def test_pagerank(nx_graph, index, alpha):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        # unweighted (karate's edges have weights), and networkx's tolerance is per node
        want = nx.pagerank(nx_graph, alpha=alpha, tol=1e-10 / len(g), weight=None)
        got = g.pagerank(alpha=alpha, tol=1e-10)
        assert got.dtype == np.float64
        np.testing.assert_allclose(got, [want[u] for u in range(len(g))], rtol=1e-6)
        # the defaults
        want = nx.pagerank(nx_graph, tol=1e-6 / len(g), weight=None)
        np.testing.assert_allclose(g.pagerank(), [want[u] for u in range(len(g))], rtol=1e-6)


def test_pagerank_engines_agree(nx_graph, index):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        auto, pull, partition = (g.pagerank(tol=1e-10, engine=e) for e in ("auto", "pull", "partition"))
        # bit for bit
        assert np.array_equal(pull, partition)
        assert np.array_equal(auto, pull)


@pytest.mark.parametrize("G", list(random_digraphs()))
def test_directed_pagerank(G):
    with sparx.DiGraph(directed(G)) as g:
        want = nx.pagerank(G, tol=1e-10 / len(g), weight=None)
        got = g.pagerank(tol=1e-10)
        np.testing.assert_allclose(got, [want[u] for u in range(len(g))], rtol=1e-6)
        assert np.array_equal(g.pagerank(tol=1e-10, engine="partition"), got)


def test_pagerank_of_multigraphs_counts_parallel_arcs():
    # parallel arcs, self-loops, and dangling nodes
    M = nx.MultiDiGraph()
    M.add_nodes_from(range(6))
    M.add_edges_from([(0, 1), (0, 1), (0, 2), (1, 2), (2, 0), (2, 2), (3, 0), (3, 3)])
    with sparx.DiGraph(directed(M)) as g:
        want = nx.pagerank(M, tol=1e-12 / len(g), weight=None)
        np.testing.assert_allclose(g.pagerank(tol=1e-12), [want[u] for u in range(len(g))], rtol=1e-8)


def test_pagerank_arguments():
    with sparx.Graph(undirected(nx.karate_club_graph())) as g:
        for kwargs in [{"alpha": 1.5}, {"alpha": -0.1}, {"tol": 0}, {"max_iter": 0}, {"engine": "push"}]:
            with pytest.raises(ValueError):
                g.pagerank(**kwargs)
    with sparx.DiGraph(directed(nx.gnp_random_graph(10, 0.3, seed=0, directed=True))) as d:
        with pytest.raises(ValueError, match="undirected"):
            d.pagerank(engine="pull")
        with pytest.raises(ValueError):
            d.pagerank(engine="other")
    with sparx.Graph(undirected(nx.karate_club_graph())) as g:
        with pytest.raises(RuntimeError, match="didn't converge"):
            g.pagerank(tol=1e-15, max_iter=2)
        assert abs(g.pagerank().sum() - 1) < 1e-9


def test_biconnectivity(nx_graph, index):
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        want = sorted(sorted(c) for c in nx.biconnected_components(nx_graph))
        got = [c.tolist() for c in g.biconnected_components()]
        assert got == want  # both by their smallest node
        b = g.biconnectivity()
        # the per node form holds the same components: each with its head
        rebuilt = [{u for u in range(len(g)) if b["components"][u] == c} | {h} for c, h in enumerate(b["heads"])]
        assert sorted(sorted(c) for c in rebuilt) == want
        assert b["articulation_points"].tolist() == sorted(nx.articulation_points(nx_graph))
        assert b["bridges"].shape == (len(list(nx.bridges(nx_graph))), 2)
        assert [tuple(e) for e in b["bridges"].tolist()] == sorted((min(e), max(e)) for e in nx.bridges(nx_graph))
        # 2-edge-connected components, numbered by their smallest node
        parts = sorted(sorted(c) for c in nx.k_edge_components(nx_graph, k=2))
        labels = b["two_edge_components"]
        assert b["two_edge_count"] == len(parts)
        assert sorted(sorted(np.flatnonzero(labels == c).tolist()) for c in range(len(parts))) == parts
        assert [labels[p[0]] for p in parts] == list(range(len(parts)))


def test_biconnectivity_of_multigraphs():
    # parallel edges aren't bridges, and change neither components nor articulation points
    edges = [(0, 1), (0, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 5)]
    path = write_mtx("multi_bcc.mtx", 7, [(max(e), min(e)) for e in edges], True)
    simple = nx.Graph(edges)
    simple.add_node(6)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    with sparx.Graph(path) as g:
        b = g.biconnectivity()
        assert [c.tolist() for c in g.biconnected_components()] == sorted(sorted(c) for c in nx.biconnected_components(simple))
        assert b["articulation_points"].tolist() == sorted(nx.articulation_points(simple))
        assert [tuple(e) for e in b["bridges"].tolist()] == [(3, 4), (4, 5)]
        assert b["components"][6] == -1  # isolated
    # only undirected graphs
    assert not hasattr(sparx.DiGraph, "biconnectivity")
