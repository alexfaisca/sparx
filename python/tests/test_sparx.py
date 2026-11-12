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


def spearman_of_pairs(pairs):
    from scipy.stats import spearmanr

    x, y = zip(*pairs)
    return spearmanr(x, y).statistic


def test_degree_assortativity(nx_graph, index):
    path = undirected(nx_graph)
    with sparx.Graph(path, index=index) as g:
        assert g.degree_assortativity() == pytest.approx(nx.degree_assortativity_coefficient(nx_graph), abs=1e-9)
        pairs = list(nx.node_degree_xy(nx_graph))
        assert g.degree_assortativity(method="spearman") == pytest.approx(spearman_of_pairs(pairs), abs=1e-9)
        with pytest.raises(ValueError):
            g.degree_assortativity(method="kendall")


def test_degree_assortativity_of_multigraphs():
    # parallel edges count each, self-loops once (and twice in their node's degree), as in a
    # networkx MultiGraph
    edges = [(0, 1), (0, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 5), (2, 2), (0, 6)]
    M = nx.MultiGraph(edges)
    path = write_mtx("multi_assort.mtx", 7, [(max(e), min(e)) for e in edges], True)
    with sparx.Graph(path) as g:
        assert g.degree_assortativity() == pytest.approx(nx.degree_assortativity_coefficient(M), abs=1e-9)
        values = np.arange(7) * 0.5
        nx.set_node_attributes(M, {u: values[u] for u in M}, "value")
        assert g.numeric_assortativity(values) == pytest.approx(nx.numeric_assortativity_coefficient(M, "value"), abs=1e-9)


def test_directed_degree_assortativity(digraph):
    with sparx.DiGraph(directed(digraph)) as d:
        for x, y in itertools.product(["out", "in"], repeat=2):
            expected = nx.degree_assortativity_coefficient(digraph, x=x, y=y)
            assert d.degree_assortativity(x=x, y=y) == pytest.approx(expected, abs=1e-9)
            pairs = list(nx.node_degree_xy(digraph, x=x, y=y))
            assert d.degree_assortativity(x=x, y=y, method="spearman") == pytest.approx(spearman_of_pairs(pairs), abs=1e-9)
        with pytest.raises(ValueError):
            d.degree_assortativity(x="total")


def test_attribute_and_numeric_assortativity(nx_graph, digraph):
    for G, cls, write in [(nx_graph, sparx.Graph, undirected), (digraph, sparx.DiGraph, directed)]:
        n = G.number_of_nodes()
        labels = [(u * 7) % 3 for u in range(n)]
        values = [((u * 37) % 11) * 0.25 - 1 for u in range(n)]
        nx.set_node_attributes(G, dict(enumerate(labels)), "label")
        nx.set_node_attributes(G, dict(enumerate(values)), "value")
        with cls(write(G)) as g:
            assert g.attribute_assortativity(labels) == pytest.approx(nx.attribute_assortativity_coefficient(G, "label"), abs=1e-9)
            # any integer labels, negative ones included, and lists
            assert g.attribute_assortativity(np.array(labels) * -1000) == pytest.approx(g.attribute_assortativity(labels))
            assert g.numeric_assortativity(values) == pytest.approx(nx.numeric_assortativity_coefficient(G, "value"), abs=1e-9)
            with pytest.raises(RuntimeError):
                g.numeric_assortativity(values[:-1])


def test_degree_assortativity_of_regular_graphs_is_nan():
    with sparx.Graph(undirected(nx.cycle_graph(6))) as g:
        assert np.isnan(g.degree_assortativity())


@pytest.mark.skipif(not sparx.GPU, reason="built without GPU support")
def test_degree_assortativity_gpu(nx_graph, digraph):
    with sparx.Graph(undirected(nx_graph)) as g:
        for method in ["pearson", "spearman"]:
            assert g.degree_assortativity(method=method, gpu=True) == g.degree_assortativity(method=method)
    with sparx.DiGraph(directed(digraph)) as d:
        assert d.degree_assortativity(x="in", y="out", gpu=True) == d.degree_assortativity(x="in", y="out")


def communities_connected(G, labels):
    for c in set(labels):
        nodes = [u for u in G if labels[u] == c]
        sub = G.subgraph(nodes)
        if not (nx.is_weakly_connected(sub) if G.is_directed() else nx.is_connected(sub)):
            return False
    return True


@pytest.mark.parametrize("method", ["louvain", "leiden"])
def test_communities_are_connected(nx_graph, digraph, method):
    for G, cls, write in [(nx_graph, sparx.Graph, undirected), (digraph, sparx.DiGraph, directed)]:
        n = G.number_of_nodes()
        with cls(write(G)) as g:
            result = getattr(g, method)()
            labels = result["communities"].tolist()
            assert communities_connected(G, labels)
            groups = [{u for u in range(n) if labels[u] == c} for c in range(result["count"])]
            assert result["modularity"] == pytest.approx(nx.community.modularity(G, groups, weight=None), abs=1e-9)


# --- weighted graphs --------------------------------------------------------------------------


def write_weighted_mtx(path, n, edges, symmetric, field="real"):
    with open(path, "w") as f:
        kind = "symmetric" if symmetric else "general"
        f.write(f"%%MatrixMarket matrix coordinate {field} {kind}\n{n} {n} {len(edges)}\n")
        for u, v, w in edges:
            f.write(f"{u + 1} {v + 1} {w}\n")
    return str(path)


def weighted_undirected(G, name="gw.mtx"):
    edges = [(max(u, v), min(u, v), w) for u, v, w in G.edges(data="weight")]
    return write_weighted_mtx(name, G.number_of_nodes(), edges, True)


def weighted_directed(D, name="dw.mtx"):
    return write_weighted_mtx(name, D.number_of_nodes(), list(D.edges(data="weight")), False)


def with_random_weights(G, seed, integers=False):
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]["weight"] = int(rng.integers(1, 10)) if integers else float(rng.uniform(0.1, 5))
    return G


def weighted_graphs():
    # karate's edges have (integer) weights of their own
    yield nx.convert_node_labels_to_integers(nx.karate_club_graph())
    yield with_random_weights(nx.gnp_random_graph(60, 0.12, seed=1), 2)


def weighted_digraphs():
    for seed in range(3):
        yield with_random_weights(nx.gnp_random_graph(40, 0.12, seed=seed, directed=True), seed, True)


@pytest.mark.parametrize("G", list(weighted_graphs()))
def test_weighted_csr(G, index):
    with sparx.Graph(weighted_undirected(G), index=index, weighted=True) as g:
        assert g.is_weighted
        src, dst = edge_list(g)
        weights = g.edge_weights()
        assert weights.dtype == np.float64 and len(weights) == g.num_edges
        for u, v, w in zip(src.tolist(), dst.tolist(), weights.tolist()):
            assert w == pytest.approx(G[u][v]["weight"])
        for u in G:
            assert g.weights(u).tolist() == pytest.approx([G[u][v]["weight"] for v in g.neighbours(u)])
    # the same file, unweighted: weights are ones
    with sparx.Graph(weighted_undirected(G, "again.mtx")) as g:
        assert not g.is_weighted
        assert g.edge_weights().tolist() == [1.0] * g.num_edges


@pytest.mark.parametrize("G", list(weighted_graphs()))
def test_weighted_algorithms_against_networkx(G):
    n = len(G)
    with sparx.Graph(weighted_undirected(G), weighted=True) as g:
        want = nx.pagerank(G, tol=1e-12 / n, weight="weight")
        np.testing.assert_allclose(g.pagerank(tol=1e-12), [want[u] for u in range(n)], rtol=1e-6)

        lv = g.louvain()
        communities = lv["communities"]
        parts = [set(np.flatnonzero(communities == c).tolist()) for c in np.unique(communities)]
        assert lv["modularity"] == pytest.approx(
            nx.community.modularity(G, parts, weight="weight"), abs=1e-9
        )
        best = nx.community.modularity(
            G, nx.community.louvain_communities(G, weight="weight", seed=0), weight="weight"
        )
        assert lv["modularity"] >= best - 0.05

        cc = g.clustering()
        local = nx.clustering(G, weight="weight")
        np.testing.assert_allclose(cc["local"], [local[u] for u in range(n)], atol=1e-12)
        assert cc["average"] == pytest.approx(nx.average_clustering(G, weight="weight"))
        # transitivity, as networkx's, ignores weights
        assert cc["transitivity"] == pytest.approx(nx.transitivity(G))

        assert g.degree_assortativity() == pytest.approx(
            nx.degree_assortativity_coefficient(G, weight="weight"), abs=1e-9
        )


@pytest.mark.parametrize("D", list(weighted_digraphs()))
def test_weighted_directed_algorithms_against_networkx(D):
    n = len(D)
    with sparx.DiGraph(weighted_directed(D), weighted=True) as g:
        want = nx.pagerank(D, tol=1e-12 / n, weight="weight")
        np.testing.assert_allclose(g.pagerank(tol=1e-12), [want[u] for u in range(n)], rtol=1e-6)

        lv = g.louvain()
        communities = lv["communities"]
        parts = [set(np.flatnonzero(communities == c).tolist()) for c in np.unique(communities)]
        assert lv["modularity"] == pytest.approx(
            nx.community.modularity(D, parts, weight="weight"), abs=1e-9
        )

        c = g.clustering()
        want = nx.clustering(D, weight="weight")
        np.testing.assert_allclose(c["local"], [want[u] for u in range(n)], atol=1e-12)

        for x, y in itertools.product(["out", "in"], repeat=2):
            assert g.degree_assortativity(x=x, y=y) == pytest.approx(
                nx.degree_assortativity_coefficient(D, x=x, y=y, weight="weight"), abs=1e-9
            )


def test_weighted_to_undirected_merges():
    path = write_weighted_mtx("m.mtx", 3, [(0, 1, 3), (1, 0, 5), (1, 2, 7)], False)
    with sparx.DiGraph(path, weighted=True) as d:
        for merge, w01 in [("sum", 8), ("min", 3), ("max", 5)]:
            u = d.to_undirected(merge=merge)
            assert u.is_weighted
            # the edge weighs the same from both ends
            assert u.weights(0).tolist() == [w01]
            assert sorted(u.weights(1).tolist()) == sorted([w01, 7])
        with pytest.raises(ValueError, match="merge"):
            d.to_undirected(merge="mean")


def test_weight_parsing_and_formats(cache_dir):
    path = write_weighted_mtx("bad.mtx", 2, [(0, 1, "abc")], False)
    # fast parsing trusts the values; checked parsing refuses
    with sparx.DiGraph(path, weighted=True) as g:
        assert np.isnan(g.edge_weights()[0])
    with pytest.raises(RuntimeError, match="abc"):
        sparx.DiGraph(path, weighted=True, weight_parsing="checked")
    with pytest.raises(ValueError, match="weight parsing"):
        sparx.DiGraph(path, weighted=True, weight_parsing="slow")
    # ggcat graphs carry no weights
    with pytest.raises(RuntimeError, match="no edge weights"):
        sparx.Graph(FIXTURES / "random_graph_1_5.lz4", weighted=True)
    # a weighted cache entry reopens weighted (ones, reopened unweighted)
    good = write_weighted_mtx("good.mtx", 2, [(0, 1, 2.5)], False)
    with sparx.DiGraph(good, weighted=True, keep_cache=True) as g:
        cache_file = g.cache_file
    with sparx.DiGraph.open(cache_file, weighted=True) as g:
        assert g.is_weighted and g.edge_weights().tolist() == [2.5]
    with sparx.DiGraph.open(cache_file) as g:
        assert g.edge_weights().tolist() == [1.0]
    for f in cache_dir.iterdir():
        f.unlink()
    # an unweighted entry doesn't reopen weighted
    with sparx.DiGraph(good, keep_cache=True) as g:
        cache_file = g.cache_file
    with pytest.raises(RuntimeError):
        sparx.DiGraph.open(cache_file, weighted=True)
    for f in cache_dir.iterdir():
        f.unlink()


# --- weighted shortest paths --------------------------------------------------------------------


def with_zero_and_real_weights(G, seed):
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        # a few zero-weight edges: nodes at distance zero
        G[u][v]["weight"] = 0.0 if rng.random() < 0.05 else float(rng.uniform(0.1, 5))
    return G


def path_lengths(G, source):
    return nx.single_source_dijkstra_path_length(G, source, weight="weight")


@pytest.mark.parametrize("seed", range(3))
def test_weighted_shortest_paths_against_networkx(seed, index):
    G = with_zero_and_real_weights(nx.gnp_random_graph(80, 0.05, seed=seed), seed)
    n = len(G)
    with sparx.Graph(weighted_undirected(G), index=index, weighted=True) as g:
        for source in [0, 17, 79]:
            want = path_lengths(G, source)
            got = g.dijkstra(source)
            assert got.dtype == np.float64
            np.testing.assert_allclose(got, [want.get(u, np.inf) for u in range(n)], rtol=1e-12)
        closeness = nx.closeness_centrality(G, distance="weight")
        reach = np.array([len(path_lengths(G, u)) for u in range(n)])
        np.testing.assert_allclose(
            g.closeness() * (reach - 1) / (n - 1), [closeness[u] for u in range(n)], rtol=1e-9
        )
        harmonic = nx.harmonic_centrality(G, distance="weight")
        np.testing.assert_allclose(
            g.harmonic(normalized=False) * (n - 1), [harmonic[u] for u in range(n)], rtol=1e-9
        )


@pytest.mark.parametrize("seed", range(3))
def test_weighted_directed_shortest_paths_against_networkx(seed):
    D = with_zero_and_real_weights(nx.gnp_random_graph(60, 0.06, seed=seed, directed=True), seed)
    n = len(D)
    with sparx.DiGraph(weighted_directed(D), weighted=True) as g:
        R = D.reverse()
        # "both": an edge either way, as long as the lightest arc between its ends
        U = nx.Graph()
        U.add_nodes_from(D)
        for u, v, w in D.edges(data="weight"):
            if not U.has_edge(u, v) or w < U[u][v]["weight"]:
                U.add_edge(u, v, weight=w)
        for source in [0, 31]:
            for direction, H in [("out", D), ("in", R), ("both", U)]:
                want = path_lengths(H, source)
                got = g.dijkstra(source, direction=direction)
                np.testing.assert_allclose(got, [want.get(u, np.inf) for u in range(n)], rtol=1e-12)
        # networkx's closeness and harmonic centralities of digraphs use distances into nodes
        closeness = nx.closeness_centrality(D, distance="weight")
        reach = np.array([len(path_lengths(R, u)) for u in range(n)])
        np.testing.assert_allclose(
            g.closeness(direction="in") * (reach - 1) / (n - 1),
            [closeness[u] for u in range(n)],
            rtol=1e-9,
        )
        harmonic = nx.harmonic_centrality(D, distance="weight")
        np.testing.assert_allclose(
            g.harmonic(normalized=False, direction="in") * (n - 1),
            [harmonic[u] for u in range(n)],
            rtol=1e-9,
        )


def test_dijkstra_on_unweighted_graphs_counts_hops():
    G = nx.gnp_random_graph(50, 0.06, seed=4)
    with sparx.Graph(undirected(G)) as g:
        hops = g.bfs(3)
        assert g.dijkstra(3).tolist() == [float(h) if h >= 0 else np.inf for h in hops.tolist()]


def test_negative_weights_fail_shortest_paths():
    path = write_weighted_mtx("neg.mtx", 3, [(0, 1, 1.0), (1, 2, -2.0)], False)
    with sparx.DiGraph(path, weighted=True) as g:
        with pytest.raises(RuntimeError, match="non-negative"):
            g.dijkstra(0)
        with pytest.raises(RuntimeError, match="non-negative"):
            g.closeness()


# --- shortest paths -------------------------------------------------------------------------


def path_length(G, nodes):
    # the lightest arc between consecutive nodes
    return sum(
        min(d["weight"] for d in G.get_edge_data(u, v, default={}).values())
        if G.is_multigraph()
        else G[u][v]["weight"]
        for u, v in zip(nodes, nodes[1:])
    )


@pytest.mark.parametrize("directed", [False, True])
def test_dijkstra_engines_and_predecessors(directed):
    G = with_random_weights(nx.gnp_random_graph(300, 0.02, seed=3, directed=directed), 5)
    make = sparx.DiGraph if directed else sparx.Graph
    path = weighted_directed(G) if directed else weighted_undirected(G)
    with make(path, weighted=True, threads=4) as g:
        for source in [0, 17]:
            want = nx.single_source_dijkstra_path_length(G, source)
            got = g.dijkstra(source)
            np.testing.assert_allclose(got, [want.get(v, np.inf) for v in range(len(g))], rtol=1e-12)
            for engine, delta in [("sequential", None), ("delta", None), ("delta", 0.3), ("delta", 50.0)]:
                d, p = g.dijkstra(source, engine=engine, delta=delta, predecessors=True)
                assert np.array_equal(d, got)
                if engine == "sequential":
                    first = p
                # the same tree, whatever the engine
                assert np.array_equal(p, first)
            assert p[source] == -1
            for v in range(len(g)):
                if v == source or np.isinf(got[v]):
                    assert p[v] == -1
                    continue
                nodes = [v]
                while nodes[-1] != source:
                    nodes.append(int(p[nodes[-1]]))
                assert path_length(G, nodes[::-1]) == pytest.approx(got[v], rel=1e-12)
    with pytest.raises(ValueError, match="delta"):
        sparx.Graph(weighted_undirected(G if not directed else G.to_undirected(), "x.mtx"), weighted=True).dijkstra(0, delta=1.0)


@pytest.mark.parametrize("directed", [False, True])
def test_shortest_paths_against_networkx(directed):
    G = with_random_weights(nx.gnp_random_graph(200, 0.03, seed=8, directed=directed), 9)
    make = sparx.DiGraph if directed else sparx.Graph
    path = weighted_directed(G) if directed else weighted_undirected(G)
    rng = np.random.default_rng(1)
    sources, targets = rng.integers(0, 200, 60), rng.integers(0, 200, 60)
    with make(path, weighted=True) as g:
        assert g.landmarks is None
        with pytest.raises(RuntimeError, match="no landmarks"):
            g.shortest_path(0, 1, heuristic="landmarks")
        for built in [False, True]:
            if built:
                landmarks = g.build_landmarks(4)
                assert len(landmarks) == 4 and np.array_equal(g.landmarks, landmarks)
            for heuristic in ["auto", "none"] + (["landmarks"] if built else []):
                for bidirectional in [None, False, True]:
                    for paths in [True, False]:
                        kw = dict(heuristic=heuristic, bidirectional=bidirectional, paths=paths)
                        many = g.shortest_paths(sources, targets, **kw)
                        for s, t, got in zip(sources.tolist(), targets.tolist(), many):
                            one = g.shortest_path(s, t, **kw)
                            try:
                                want = nx.shortest_path_length(G, s, t, weight="weight")
                            except nx.NetworkXNoPath:
                                assert got is None and one is None
                                continue
                            assert got["distance"] == pytest.approx(want, rel=1e-12)
                            assert one["distance"] == got["distance"]
                            if paths:
                                nodes = got["nodes"].tolist()
                                assert (nodes[0], nodes[-1]) == (s, t)
                                assert path_length(G, nodes) == pytest.approx(want, rel=1e-12)
                            else:
                                assert "nodes" not in got
        g.drop_landmarks()
        assert g.landmarks is None
        with pytest.raises(ValueError, match="out of range"):
            g.shortest_paths([0, 1], [2, 200])
        with pytest.raises(ValueError, match="as many"):
            g.shortest_paths([0, 1], [2])
        with pytest.raises(ValueError, match="heuristic"):
            g.shortest_path(0, 1, heuristic="euclid")


def test_landmarks_are_kept_with_the_cache(cache_dir):
    D = with_random_weights(nx.gnp_random_graph(80, 0.06, seed=4, directed=True), 2, True)
    with sparx.DiGraph(weighted_directed(D), weighted=True, keep_cache=True) as g:
        landmarks = g.build_landmarks(3)
        cache_file = g.cache_file
    with sparx.DiGraph.open(cache_file, weighted=True) as g:
        assert np.array_equal(g.landmarks, landmarks)
        got = g.shortest_path(0, 5, heuristic="landmarks")
        assert (got is None) == (not nx.has_path(D, 0, 5))
    # an unweighted graph of the same entry isn't theirs
    with sparx.DiGraph.open(cache_file) as g:
        with pytest.raises(RuntimeError, match="another graph"):
            g.shortest_path(0, 5, heuristic="landmarks")
    for f in cache_dir.iterdir():
        f.unlink()


# --- spectral clustering --------------------------------------------------------------------


def nmi(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(joint, (ia, ib), 1)
    p = joint / n
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz])).sum()
    h = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()
    return 2 * mi / (h(pa) + h(pb))


def test_spectral_clustering_finds_planted_partitions():
    G = nx.planted_partition_graph(6, 120, 0.12, 0.006, seed=2)
    truth = [u // 120 for u in range(len(G))]
    with sparx.Graph(undirected(G), threads=4) as g:
        for precision in ["f32", "f64"]:
            s = g.spectral_clustering(6, precision=precision)
            assert s["communities"].dtype == np.int64 and s["count"] == 6
            assert nmi(s["communities"], truth) > 0.95
            parts = [set(np.flatnonzero(s["communities"] == c).tolist()) for c in range(s["count"])]
            assert s["modularity"] == pytest.approx(nx.community.modularity(G, parts), abs=1e-9)
            # numbered by their least node
            firsts = [int(np.flatnonzero(s["communities"] == c)[0]) for c in range(s["count"])]
            assert firsts == sorted(firsts)
        bethe = g.spectral_clustering(6, matrix="bethe")
        assert nmi(bethe["communities"], truth) > 0.95
        counted = g.spectral_clustering(None, matrix="bethe", moments=512, probes=64)
        assert abs(counted["k_estimate"] - 6) <= 3 * counted["k_error"]
    # the same whatever the threads
    with sparx.Graph(undirected(G, "again.mtx"), threads=1) as g1:
        with sparx.Graph(undirected(G, "again2.mtx"), threads=7) as g7:
            assert np.array_equal(g1.spectral_clustering(6)["communities"], g7.spectral_clustering(6)["communities"])


def test_spectral_clustering_of_weighted_graphs_and_errors():
    G = nx.gnp_random_graph(300, 0.05, seed=5)
    for u, v in G.edges():
        G[u][v]["weight"] = 10.0 if u // 100 == v // 100 else 0.3
    with sparx.Graph(weighted_undirected(G), weighted=True) as g:
        s = g.spectral_clustering(3)
        assert nmi(s["communities"], [u // 100 for u in range(300)]) > 0.9
        with pytest.raises(RuntimeError, match="unweighted"):
            g.spectral_clustering(3, matrix="bethe")
        with pytest.raises(RuntimeError, match="needs k"):
            g.spectral_clustering(None)
        with pytest.raises(ValueError, match="matrix"):
            g.spectral_clustering(3, matrix="adjacency")
        with pytest.raises(ValueError, match="precision"):
            g.spectral_clustering(3, precision="f16")
        with pytest.raises(RuntimeError, match="k must be"):
            g.spectral_clustering(301)
    with sparx.Graph(undirected(G, "rejects.mtx")) as g:
        with pytest.raises(ValueError, match="engine"):
            g.spectral_clustering(3, engine="lanczos")


def test_label_propagation():
    G = nx.planted_partition_graph(5, 150, 0.15, 0.004, seed=6)
    truth = [u // 150 for u in range(len(G))]
    found = []
    for threads in [1, 6]:
        with sparx.Graph(undirected(G, f"lpa{threads}.mtx"), threads=threads) as g:
            lp = g.label_propagation()
            assert lp["communities"].dtype == np.int64 and lp["count"] == len(np.unique(lp["communities"]))
            assert nmi(lp["communities"], truth) > 0.95
            parts = [set(np.flatnonzero(lp["communities"] == c).tolist()) for c in range(lp["count"])]
            assert lp["modularity"] == pytest.approx(nx.community.modularity(G, parts), abs=1e-9)
            assert all(nx.is_connected(G.subgraph(p)) for p in parts)
            assert lp["moves"][-1] <= 0.05 * len(G) and lp["colors"] >= 2
            found.append(lp["communities"])
    assert np.array_equal(found[0], found[1])
    with sparx.Graph(undirected(G, "lpa_bad.mtx")) as g:
        with pytest.raises(RuntimeError, match="tolerance"):
            g.label_propagation(tolerance=2.0)
    with sparx.Graph(undirected(G, "lpa_direction.mtx")) as g:
        with pytest.raises(ValueError, match="DiGraph"):
            g.label_propagation(direction="in")


def directed_planted(seed=8):
    G = nx.planted_partition_graph(4, 120, 0.15, 0.004, seed=seed)
    rng = np.random.default_rng(seed)
    D = nx.DiGraph()
    D.add_nodes_from(G)
    D.add_edges_from((u, v) if rng.random() < 0.5 else (v, u) for u, v in G.edges())
    return D, [u // 120 for u in range(len(D))]


def test_directed_label_propagation():
    D, truth = directed_planted()
    with sparx.DiGraph(directed(D)) as d:
        both = d.label_propagation()
        with d.to_undirected(mode="each") as u:
            assert np.array_equal(both["communities"], u.label_propagation()["communities"])
        for direction in ["both", "in", "out"]:
            lp = d.label_propagation(direction=direction)
            assert nmi(lp["communities"], truth) > 0.95, direction
            parts = [set(np.flatnonzero(lp["communities"] == c).tolist()) for c in range(lp["count"])]
            assert lp["modularity"] == pytest.approx(nx.community.modularity(D, parts), abs=1e-9)
        with pytest.raises(ValueError, match="direction"):
            d.label_propagation(direction="sideways")


def regularized_laplacian(G):
    A = nx.to_numpy_array(G, nodelist=range(len(G)))
    d = A.sum(1)
    s = 1 / np.sqrt(d + d.mean())
    return np.eye(len(G)) - s[:, None] * A * s[None, :]


def test_spectral_embedding_and_exact_engine():
    G = nx.planted_partition_graph(5, 60, 0.25, 0.01, seed=9)
    truth = [u // 60 for u in range(len(G))]
    want = np.linalg.eigvalsh(regularized_laplacian(G))
    with sparx.Graph(undirected(G)) as g:
        e = g.spectral_embedding(8, tolerance=1e-9)
        assert e["converged"] and e["vectors"].shape == (len(G), 8) and e["vectors"].dtype == np.float64
        np.testing.assert_allclose(e["eigenvalues"], want[:8], atol=1e-7)
        np.testing.assert_allclose(e["vectors"].T @ e["vectors"], np.eye(8), atol=1e-8)
        assert g.spectral_embedding(8, precision="f32")["vectors"].dtype == np.float32
        given = g.spectral_clustering(5, engine="exact")
        assert given["converged"] and nmi(given["communities"], truth) > 0.95
        np.testing.assert_allclose(given["eigenvalues"], want[:5], atol=1e-4)
        chosen = g.spectral_clustering(None, engine="exact", max_k=12)
        assert chosen["count"] == 5 and len(chosen["eigenvalues"]) == 13
        with pytest.raises(RuntimeError, match="max_k"):
            g.spectral_clustering(None, engine="exact")
    D, dtruth = directed_planted(10)
    with sparx.DiGraph(directed(D)) as d:
        for engine in ["compressive", "exact"]:
            s = d.spectral_clustering(4, engine=engine)
            with d.to_undirected(mode="each") as u:
                assert np.array_equal(s["communities"], u.spectral_clustering(4, engine=engine)["communities"])
            assert nmi(s["communities"], dtruth) > 0.95
        assert d.spectral_embedding(4)["converged"]


def test_f32_weights_and_to_directed_keep_weights(cache_dir):
    G = with_random_weights(nx.gnp_random_graph(80, 0.08, seed=12), 3)
    path = weighted_undirected(G)
    with sparx.Graph(path, weighted=True) as g64, sparx.Graph(path, weighted=True, weight_type="f32", keep_cache=True) as g32:
        assert (g64.weight_type, g32.weight_type) == ("f64", "f32")
        np.testing.assert_allclose(g32.edge_weights(), g64.edge_weights(), rtol=1e-6)
        np.testing.assert_allclose(g32.pagerank(tol=1e-10), g64.pagerank(tol=1e-10), rtol=1e-5)
        cache_file = g32.cache_file
        # weights survive to_directed
        d = g64.to_directed()
        assert d.is_weighted and d.weight_type == "f64"
        assert np.array_equal(d.edge_weights(), g64.edge_weights())
    with sparx.Graph.open(cache_file, weighted=True, weight_type="f32") as g:
        assert g.weight_type == "f32"
    with pytest.raises(ValueError, match="weight type"):
        sparx.Graph(path, weighted=True, weight_type="f16")
    for f in cache_dir.iterdir():
        f.unlink()


def test_modularity_of_given_partitions():
    G = with_random_weights(nx.planted_partition_graph(3, 30, 0.3, 0.02, seed=4), 5)
    labels = np.array([u // 30 for u in range(len(G))])
    parts = [set(range(c * 30, (c + 1) * 30)) for c in range(3)]
    with sparx.Graph(undirected(G)) as g:
        assert g.modularity(labels) == pytest.approx(nx.community.modularity(G, parts, weight=None), abs=1e-12)
    with sparx.Graph(weighted_undirected(G), weighted=True) as g:
        assert g.modularity(labels) == pytest.approx(nx.community.modularity(G, parts), abs=1e-12)
        with pytest.raises(ValueError, match="labels for"):
            g.modularity(labels[:-1])
        with pytest.raises(ValueError, match="out of range"):
            g.modularity(np.full(len(G), len(G)))
    D, truth = directed_planted()
    dparts = [set(np.flatnonzero(np.array(truth) == c).tolist()) for c in range(4)]
    with sparx.DiGraph(directed(D)) as d:
        assert d.modularity(np.array(truth)) == pytest.approx(nx.community.modularity(D, dparts), abs=1e-12)


def test_local_communities():
    G = nx.planted_partition_graph(4, 50, 0.3, 0.005, seed=13)
    with sparx.Graph(undirected(G)) as g:
        for c in [g.hk_relax([3], t=10.0, eps=1e-4), g.hkpr(3, eps=0.1, target_size=60, target_volume=800)]:
            nodes = set(c["nodes"].tolist())
            assert 3 in nodes and c["size"] == len(nodes)
            assert c["conductance"] == pytest.approx(nx.conductance(G, nodes), abs=1e-12)
            assert c["volume"] == pytest.approx(nx.volume(G, nodes))
        # seeded: the same walks, the same community
        a, b = g.hkpr(3, random_seed=1), g.hkpr(3, random_seed=1)
        assert np.array_equal(a["nodes"], b["nodes"]) and np.array_equal(a["heat"], b["heat"])
        # the sweep's community is the best prefix by conductance (up to twice the target
        # size), whatever order the diffusion comes in
        heat = np.array([2.0 if u < 50 else 1.0 / (u + 1) for u in range(200)])
        order = np.argsort(-heat, kind="stable")
        best = min((nx.conductance(G, set(order[:i].tolist())), i) for i in range(1, 101))
        cut = g.sweep_cut(np.arange(200)[::-1], heat[::-1], target_size=50)
        assert cut["conductance"] == pytest.approx(best[0], abs=1e-12)
        assert set(cut["nodes"].tolist()) == set(order[: best[1]].tolist())
        with pytest.raises(ValueError, match="as many"):
            g.sweep_cut([0, 1], [1.0])
        with pytest.raises(ValueError, match="out of range"):
            g.sweep_cut([0, 500], [1.0, 0.5])
    assert not hasattr(sparx.DiGraph, "hk_relax")


@pytest.mark.parametrize("G", list(weighted_graphs()))
def test_f32_weights(G):
    with sparx.Graph(weighted_undirected(G), weighted=True, weight_type="f32") as g:
        assert g.weight_type == "f32"
        src, dst = edge_list(g)
        want = [np.float32(G[u][v]["weight"]) for u, v in zip(src.tolist(), dst.tolist())]
        np.testing.assert_array_equal(g.edge_weights(), np.array(want, dtype=np.float64))
    with sparx.Graph(weighted_undirected(G), weighted=True) as g:
        assert g.weight_type == "f64"
    with sparx.Graph(undirected(G)) as g:
        assert g.weight_type is None
    with pytest.raises(ValueError):
        sparx.Graph(weighted_undirected(G), weighted=True, weight_type="f16")


def test_f32_weighted_graphs_reopen(cache_dir):
    G = with_random_weights(nx.gnp_random_graph(30, 0.2, seed=4), 4)
    with sparx.Graph(weighted_undirected(G), weighted=True, weight_type="f32", keep_cache=True) as g:
        cache_file, weights = g.cache_file, g.edge_weights()
    with sparx.Graph.open(cache_file, weighted=True, weight_type="f32") as g:
        assert g.weight_type == "f32"
        np.testing.assert_array_equal(g.edge_weights(), weights)
    for f in os.listdir(cache_dir):
        os.remove(cache_dir / f)


@pytest.mark.parametrize("weighted", [False, True])
def test_modularity_of_given_partitions_against_networkx(weighted):
    G = with_random_weights(nx.gnp_random_graph(50, 0.1, seed=3), 3)
    D = with_random_weights(nx.gnp_random_graph(40, 0.12, seed=3, directed=True), 3, True)
    labels = np.random.default_rng(0).integers(0, 5, 50)
    weight = "weight" if weighted else None
    path = weighted_undirected(G) if weighted else undirected(G)
    with sparx.Graph(path, weighted=weighted) as g:
        parts = [set(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)]
        assert g.modularity(labels) == pytest.approx(
            nx.community.modularity(G, parts, weight=weight), abs=1e-12
        )
        with pytest.raises(ValueError):
            g.modularity(labels[:-1])
        with pytest.raises(ValueError):
            g.modularity(np.full(50, -1))
    path = weighted_directed(D) if weighted else directed(D)
    with sparx.DiGraph(path, weighted=weighted) as d:
        labels = labels[:40]
        parts = [set(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)]
        assert d.modularity(labels) == pytest.approx(
            nx.community.modularity(D, parts, weight=weight), abs=1e-12
        )


def conductance(G, nodes):
    nodes = set(nodes)
    cut = nx.cut_size(G, nodes)
    return cut / min(nx.volume(G, nodes), nx.volume(G, set(G) - nodes))


def two_cliques():
    # two 8-cliques joined by one edge: the community around a node is its clique
    G = nx.disjoint_union(nx.complete_graph(8), nx.complete_graph(8))
    G.add_edge(0, 8)
    return G


def test_local_communities():
    G = two_cliques()
    with sparx.Graph(undirected(G)) as g:
        for community in [g.hk_relax([3]), g.hkpr(3), g.hkpr(3, random_seed=7)]:
            assert set(community["nodes"].tolist()) == set(range(8))
            assert community["size"] == 8
            assert community["volume"] == nx.volume(G, range(8))
            assert community["conductance"] == pytest.approx(conductance(G, range(8)))
            assert np.all(np.diff(community["heat"]) <= 0)
        # the same seed, the same community
        assert g.hkpr(12)["nodes"].tolist() == g.hkpr(12)["nodes"].tolist()
        with pytest.raises(ValueError):
            g.hk_relax([16])
        with pytest.raises(ValueError):
            g.hkpr(16)


def test_sweep_cut_finds_the_best_prefix():
    G = two_cliques()
    rng = np.random.default_rng(5)
    with sparx.Graph(undirected(G)) as g:
        nodes = rng.permutation(16)
        heat = rng.uniform(0, 1, 16)
        order = sorted(range(16), key=lambda i: (-heat[i], nodes[i]))
        prefixes = [nodes[order[: k + 1]].tolist() for k in range(15)]
        best = min(conductance(G, p) for p in prefixes)
        assert g.sweep_cut(nodes, heat)["conductance"] == pytest.approx(best)
        # heat that's high on one clique cuts it out
        heat = np.where(np.arange(16) < 8, 1.0, 0.0) + rng.uniform(0, 0.1, 16)
        assert set(g.sweep_cut(np.arange(16), heat)["nodes"].tolist()) == set(range(8))
        with pytest.raises(ValueError):
            g.sweep_cut([0, 1], [1.0])
        with pytest.raises(ValueError):
            g.sweep_cut([0, 16], [1.0, 0.5])
        with pytest.raises(ValueError):
            g.sweep_cut([0, 1], [1.0, np.nan])


def test_betweenness_against_networkx(nx_graph, index):
    want = nx.betweenness_centrality(nx_graph)
    raw = nx.betweenness_centrality(nx_graph, normalized=False)
    with sparx.Graph(undirected(nx_graph), index=index) as g:
        b = g.betweenness(engine="exact")
        assert b["exact"] and b["samples"] == 0 and "top" not in b
        np.testing.assert_allclose(b["betweenness"], [want[u] for u in range(len(nx_graph))], atol=1e-12)
        b = g.betweenness(engine="exact", normalized=False)["betweenness"]
        np.testing.assert_allclose(b, [raw[u] for u in range(len(nx_graph))], atol=1e-9)
        # small graphs are exact by default
        assert g.betweenness()["exact"]


@pytest.mark.parametrize("seed", range(3))
def test_directed_betweenness_against_networkx(seed):
    D = nx.gnp_random_graph(40, 0.08, seed=seed, directed=True)
    want = nx.betweenness_centrality(D)
    with sparx.DiGraph(directed(D)) as d:
        b = d.betweenness(engine="exact")["betweenness"]
        np.testing.assert_allclose(b, [want[u] for u in range(40)], atol=1e-12)


@pytest.mark.parametrize("G", list(weighted_graphs()))
def test_weighted_betweenness_against_networkx(G):
    want = nx.betweenness_centrality(G, weight="weight")
    with sparx.Graph(weighted_undirected(G), weighted=True) as g:
        b = g.betweenness(engine="exact")["betweenness"]
        np.testing.assert_allclose(b, [want[u] for u in range(len(G))], atol=1e-12)
        approx = g.betweenness(engine="kadabra", epsilon=0.02)
        n = len(G)
        assert np.max(np.abs(approx["betweenness"] - b)) <= 0.02 * n / (n - 2)


@pytest.mark.parametrize("D", list(weighted_digraphs()))
def test_weighted_directed_betweenness_against_networkx(D):
    want = nx.betweenness_centrality(D, weight="weight")
    with sparx.DiGraph(weighted_directed(D), weighted=True) as d:
        b = d.betweenness(engine="exact")["betweenness"]
        np.testing.assert_allclose(b, [want[u] for u in range(len(D))], atol=1e-12)


def test_kadabra_betweenness():
    G = nx.barabasi_albert_graph(400, 3, seed=2)
    D = nx.gnp_random_graph(200, 0.03, seed=2, directed=True)
    for graph, cls, path in [(G, sparx.Graph, undirected(G)), (D, sparx.DiGraph, directed(D))]:
        n = len(graph)
        want = nx.betweenness_centrality(graph)
        want = np.array([want[u] for u in range(n)])
        runs = []
        for threads in [1, 4]:
            with cls(path, threads=threads) as g:
                b = g.betweenness(engine="kadabra", epsilon=0.01)
                assert not b["exact"] and 0 < b["samples"] <= b["sample_cap"]
                assert np.max(np.abs(b["betweenness"] - want)) <= 0.01 * n / (n - 2)
                runs.append(b)
                other = g.betweenness(engine="kadabra", epsilon=0.01, seed=1)["betweenness"]
                assert not np.array_equal(other, b["betweenness"])
        # the same whatever the threads
        assert runs[0]["samples"] == runs[1]["samples"]
        np.testing.assert_array_equal(runs[0]["betweenness"], runs[1]["betweenness"])


def test_betweenness_top_k():
    G = nx.barabasi_albert_graph(500, 2, seed=5)
    want = nx.betweenness_centrality(G)
    best = max(want, key=want.get)
    with sparx.Graph(undirected(G)) as g:
        exact = g.betweenness(engine="exact", k=5)
        assert exact["top"].tolist() == sorted(want, key=lambda u: (-want[u], u))[:5]
        top = g.betweenness(engine="kadabra", epsilon=0.001, k=1)
        assert top["top"].tolist() == [best]
        assert top["samples"] < top["sample_cap"]


def test_betweenness_invalid_arguments():
    G = nx.path_graph(4)
    with sparx.Graph(undirected(G)) as g:
        for kwargs in [
            {"engine": "fast"},
            {"epsilon": 0},
            {"epsilon": 1.5},
            {"delta": 0},
            {"k": 0},
        ]:
            with pytest.raises(ValueError):
                g.betweenness(**kwargs)
    G = nx.path_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.0 if u == 1 else 1.0
    with sparx.Graph(weighted_undirected(G), weighted=True) as g:
        with pytest.raises(Exception):
            g.betweenness()


@pytest.mark.parametrize("seed", range(4))
def test_betweenness_on_graphs_with_articulation_points(seed):
    # trees joined into a few cycles, and cliques chained by paths: many biconnected components
    G = nx.random_labeled_tree(60, seed=seed) if hasattr(nx, "random_labeled_tree") else nx.random_tree(60, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        u, v = rng.integers(0, 60, 2)
        if u != v:
            G.add_edge(int(u), int(v))
    G = nx.disjoint_union_all([G, nx.barbell_graph(5, 4), nx.lollipop_graph(4, 3), nx.empty_graph(2)])
    want = nx.betweenness_centrality(G)
    raw = nx.betweenness_centrality(G, normalized=False)
    with sparx.Graph(undirected(G), threads=4) as g:
        b = g.betweenness(engine="exact")["betweenness"]
        np.testing.assert_allclose(b, [want[u] for u in range(len(G))], atol=1e-12)
        b = g.betweenness(engine="exact", normalized=False)["betweenness"]
        np.testing.assert_allclose(b, [raw[u] for u in range(len(G))], atol=1e-9)


def test_label_propagation_exploration_auto():
    G = nx.planted_partition_graph(5, 60, 0.3, 0.01, seed=3)
    with sparx.Graph(undirected(G)) as g:
        full = g.label_propagation()
        auto = g.label_propagation(exploration="auto")
        assert len(auto["moves"]) <= len(full["moves"])
        assert auto["count"] >= 1 and auto["modularity"] > 0.5
        # the same every run
        assert auto["communities"].tolist() == g.label_propagation(exploration="auto")["communities"].tolist()
        with pytest.raises(ValueError):
            g.label_propagation(exploration="fast")
        with pytest.raises(TypeError):
            g.label_propagation(exploration=1.5)


def test_label_propagation_asynchronous():
    G = nx.planted_partition_graph(5, 60, 0.3, 0.01, seed=3)
    truth = np.repeat(np.arange(5), 60)
    with sparx.Graph(undirected(G), threads=1) as g:
        r = g.label_propagation(engine="asynchronous")
        assert r["colors"] == 1 and r["modularity"] > 0.5
        # repeatable on one thread
        again = g.label_propagation(engine="asynchronous")
        assert r["communities"].tolist() == again["communities"].tolist()
        parts = [set(np.flatnonzero(r["communities"] == c).tolist()) for c in np.unique(r["communities"])]
        assert r["modularity"] == pytest.approx(nx.community.modularity(G, parts), abs=1e-9)
        assert g.label_propagation(engine="asynchronous", exploration="auto")["modularity"] > 0.5
        with pytest.raises(ValueError):
            g.label_propagation(engine="fast")
    D = nx.DiGraph(G)
    with sparx.DiGraph(directed(D)) as d:
        assert d.label_propagation(engine="asynchronous", direction="in")["modularity"] > 0.4
