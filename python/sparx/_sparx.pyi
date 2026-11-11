from os import PathLike
from typing import Literal, Optional, TypedDict, Union, overload

import numpy as np
import numpy.typing as npt

Width = Literal["u8", "u16", "u32", "u64"]
Index = Literal["u32", "u64"]
Direction = Literal["out", "in", "both"]
CoreDegree = Literal["in", "out", "total"]
UndirectedMode = Literal["collapse", "each", "mutual"]
WeightParsing = Literal["fast", "checked"]
WeightType = Literal["f64", "f32"]
WeightMerge = Literal["sum", "min", "max"]
Engine = Literal["auto", "sequential", "delta"]
Guide = Literal["auto", "landmarks", "none"]
NodeIds = Union[npt.ArrayLike, list[int]]

DIRECTED_TRIANGLE_TYPES: list[str]
GPU: bool

class Path(TypedDict, total=False):
    distance: float
    nodes: npt.NDArray[np.int64]  # left out with paths=False
    expanded: int

class Spectral(TypedDict, total=False):
    communities: npt.NDArray[np.int64]
    count: int
    modularity: float
    lambda_k: float
    k_estimate: float  # only when the compressive engine estimated k
    k_error: float
    eigenvalues: npt.NDArray[np.float64]  # only with the exact engine
    converged: bool

class Embedding(TypedDict):
    eigenvalues: npt.NDArray[np.float64]
    vectors: npt.NDArray[np.floating]
    residuals: npt.NDArray[np.float64]
    iterations: int
    converged: bool

class LabelPropagation(TypedDict):
    communities: npt.NDArray[np.int64]
    count: int
    modularity: float
    moves: npt.NDArray[np.int64]
    colors: int

class Community(TypedDict):
    nodes: npt.NDArray[np.int64]  # by descending heat
    heat: npt.NDArray[np.float64]
    size: int
    volume: int
    width: int
    conductance: float

class BetweennessResult(TypedDict, total=False):
    betweenness: npt.NDArray[np.float64]
    exact: bool
    samples: int
    sample_cap: int
    top: npt.NDArray[np.int64]  # only with k

class Clustering(TypedDict):
    local: npt.NDArray[np.float64]
    average: float
    transitivity: float

class DirectedClustering(TypedDict):
    local: npt.NDArray[np.float64]
    average: float

class Louvain(TypedDict):
    communities: npt.NDArray[np.int64]
    count: int
    modularity: float

class HyperBall(TypedDict):
    distances: npt.NDArray[np.float64]
    harmonic_distances: npt.NDArray[np.float64]
    closeness: npt.NDArray[np.float64]
    harmonic: npt.NDArray[np.float64]
    lin: npt.NDArray[np.float64]

class Components(TypedDict):
    components: npt.NDArray[np.int64]
    count: int
    sizes: npt.NDArray[np.int64]

class Biconnectivity(TypedDict):
    components: npt.NDArray[np.int64]
    heads: npt.NDArray[np.int64]
    articulation_points: npt.NDArray[np.int64]
    bridges: npt.NDArray[np.int64]
    two_edge_components: npt.NDArray[np.int64]
    two_edge_count: int

class Graph:
    """An undirected graph: loading a directed dataset raises `RuntimeError` (load it as a `DiGraph`)."""

    def __init__(
        self,
        path: Union[str, PathLike[str]],
        *,
        index: Index = "u64",
        threads: Optional[int] = None,
        keep_cache: bool = False,
        id: Optional[str] = None,
        weighted: bool = False,
        weight_parsing: WeightParsing = "fast",
        weight_type: WeightType = "f64",
    ) -> None: ...
    @staticmethod
    def open(
        cache_file: Union[str, PathLike[str]],
        *,
        index: Index = "u64",
        threads: Optional[int] = None,
        weighted: bool = False,
        weight_type: WeightType = "f64",
    ) -> Graph: ...
    def close(self) -> None: ...
    def __enter__(self) -> Graph: ...
    def __exit__(self, _ty: object, _value: object, _traceback: object) -> bool: ...
    def __len__(self, /) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def index(self) -> Index: ...
    @property
    def threads(self) -> int: ...
    @property
    def cache_id(self) -> str: ...
    @property
    def cache_file(self) -> str: ...
    def metadata(self) -> str: ...
    @property
    def is_directed(self) -> Literal[False]: ...
    @property
    def is_symmetric(self) -> Literal[True]: ...
    def degree(self, u: int) -> int: ...
    def neighbours(self, u: int) -> npt.NDArray[np.int64]: ...
    def csr(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
    @property
    def is_weighted(self) -> bool: ...
    def weights(self, u: int) -> npt.NDArray[np.float64]: ...
    def edge_weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def weight_type(self) -> Optional[WeightType]: ...
    def modularity(self, communities: npt.ArrayLike) -> float: ...
    def components(self, *, kind: Literal["weak", "strong"] = "weak") -> Components: ...
    def triangles(self, *, width: Width = "u8", gpu: bool = False) -> npt.NDArray[np.unsignedinteger]: ...
    def k_truss(
        self, *, algorithm: Literal["pkt", "burkhardt"] = "pkt", width: Width = "u8"
    ) -> npt.NDArray[np.unsignedinteger]: ...
    def k_core(
        self, *, algorithm: Literal["liu", "bz"] = "liu", width: Width = "u8"
    ) -> npt.NDArray[np.unsignedinteger]: ...
    def clustering(self) -> Clustering: ...
    def louvain(self) -> Louvain: ...
    def label_propagation(
        self,
        *,
        max_iterations: int = 40,
        tolerance: float = 0.05,
        exploration: int = 20,
        direction: Literal["both", "in", "out"] = "both",
    ) -> LabelPropagation: ...
    def betweenness(
        self,
        *,
        engine: Literal["auto", "exact", "kadabra"] = "auto",
        epsilon: float = 0.01,
        delta: float = 0.1,
        k: Optional[int] = None,
        normalized: bool = True,
        seed: int = 0x5EED,
    ) -> BetweennessResult: ...
    def spectral_clustering(
        self,
        k: Optional[int],
        *,
        matrix: Literal["laplacian", "bethe"] = "laplacian",
        engine: Literal["compressive", "exact"] = "compressive",
        regularization: Optional[float] = None,
        filter_degree: int = 64,
        moments: int = 128,
        probes: int = 16,
        features: Optional[int] = None,
        sample: Optional[int] = None,
        restarts: int = 4,
        refine: int = 3,
        seed: int = 0x5EED,
        max_k: int = 4096,
        subspace_degree: int = 16,
        tolerance: float = 1e-5,
        max_iterations: int = 500,
        precision: Literal["f32", "f64"] = "f32",
    ) -> Spectral: ...
    def spectral_embedding(
        self,
        count: int,
        *,
        matrix: Literal["laplacian", "bethe"] = "laplacian",
        regularization: Optional[float] = None,
        degree: int = 16,
        tolerance: float = 1e-5,
        max_iterations: int = 500,
        oversampling: Optional[int] = None,
        seed: int = 0x5EED,
        precision: Literal["f32", "f64"] = "f64",
    ) -> Embedding: ...
    def leiden(self) -> Louvain: ...
    def hyperball(self, *, precision: int = 8, normalized: bool = True, gpu: bool = False) -> HyperBall: ...
    def closeness(self, *, normalized: bool = True) -> npt.NDArray[np.float64]: ...
    def harmonic(self, *, normalized: bool = True) -> npt.NDArray[np.float64]: ...
    def lin(self) -> npt.NDArray[np.float64]: ...
    def pagerank(
        self,
        *,
        alpha: float = 0.85,
        tol: float = 1e-6,
        max_iter: int = 100,
        engine: Literal["auto", "pull", "partition"] = "auto",
    ) -> npt.NDArray[np.float64]: ...
    def biconnectivity(self) -> Biconnectivity: ...
    def biconnected_components(self) -> list[npt.NDArray[np.int64]]: ...
    def degree_assortativity(
        self, *, method: Literal["pearson", "spearman"] = "pearson", gpu: bool = False
    ) -> float: ...
    def numeric_assortativity(self, values: npt.ArrayLike) -> float: ...
    def attribute_assortativity(self, labels: npt.ArrayLike) -> float: ...
    def bfs(self, source: int) -> npt.NDArray[np.int64]: ...
    @overload
    def dijkstra(
        self, source: int, *, engine: Engine = "auto", delta: Optional[float] = None,
        predecessors: Literal[False] = False,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def dijkstra(
        self, source: int, *, engine: Engine = "auto", delta: Optional[float] = None,
        predecessors: Literal[True],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ...
    def build_landmarks(self, k: int = 8) -> npt.NDArray[np.int64]: ...
    @property
    def landmarks(self) -> Optional[npt.NDArray[np.int64]]: ...
    def drop_landmarks(self) -> None: ...
    def shortest_path(
        self,
        source: int,
        target: int,
        *,
        heuristic: Guide = "auto",
        bidirectional: Optional[bool] = None,
        paths: bool = True,
    ) -> Optional[Path]: ...
    def shortest_paths(
        self,
        sources: NodeIds,
        targets: NodeIds,
        *,
        heuristic: Guide = "auto",
        bidirectional: Optional[bool] = None,
        paths: bool = True,
    ) -> list[Optional[Path]]: ...
    def hk_relax(
        self,
        seeds: list[int],
        *,
        t: float = 10.0,
        eps: float = 1e-4,
        target_size: Optional[int] = None,
        target_volume: Optional[int] = None,
    ) -> Community: ...
    def hkpr(
        self,
        seed: int,
        *,
        eps: float = 0.1,
        target_size: Optional[int] = None,
        target_volume: Optional[int] = None,
        target_conductance: float = 0.1,
        random_seed: int = 0x5EED,
    ) -> Community: ...
    def sweep_cut(
        self,
        nodes: NodeIds,
        heat: npt.ArrayLike,
        *,
        target_size: Optional[int] = None,
        target_volume: Optional[int] = None,
    ) -> Community: ...

    def to_directed(self) -> DiGraph:
        """The graph as a `DiGraph` mapping this graph's cache entry: usable while this graph is open."""

class DiGraph:
    """A directed graph: any dataset loads as one."""

    def __init__(
        self,
        path: Union[str, PathLike[str]],
        *,
        index: Index = "u64",
        threads: Optional[int] = None,
        keep_cache: bool = False,
        id: Optional[str] = None,
        weighted: bool = False,
        weight_parsing: WeightParsing = "fast",
        weight_type: WeightType = "f64",
    ) -> None: ...
    @staticmethod
    def open(
        cache_file: Union[str, PathLike[str]],
        *,
        index: Index = "u64",
        threads: Optional[int] = None,
        weighted: bool = False,
        weight_type: WeightType = "f64",
    ) -> DiGraph: ...
    def close(self) -> None: ...
    def __enter__(self) -> DiGraph: ...
    def __exit__(self, _ty: object, _value: object, _traceback: object) -> bool: ...
    def __len__(self, /) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def index(self) -> Index: ...
    @property
    def threads(self) -> int: ...
    @property
    def cache_id(self) -> str: ...
    @property
    def cache_file(self) -> str: ...
    def metadata(self) -> str: ...
    @property
    def is_directed(self) -> Literal[True]: ...
    @property
    def is_symmetric(self) -> bool: ...
    def degree(self, u: int) -> int: ...
    def neighbours(self, u: int) -> npt.NDArray[np.int64]: ...
    def csr(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
    @property
    def is_weighted(self) -> bool: ...
    def weights(self, u: int) -> npt.NDArray[np.float64]: ...
    def edge_weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def weight_type(self) -> Optional[WeightType]: ...
    def modularity(self, communities: npt.ArrayLike) -> float: ...
    def components(self, *, kind: Literal["weak", "strong"] = "weak") -> Components: ...
    def in_degree(self, u: int) -> int: ...
    def in_neighbours(self, u: int) -> npt.NDArray[np.int64]: ...
    def directed_triangles(self, kind: str, *, width: Width = "u8") -> npt.NDArray[np.unsignedinteger]: ...
    def k_truss(
        self, kind: str, *, algorithm: Literal["pkt", "burkhardt"] = "pkt", width: Width = "u8"
    ) -> npt.NDArray[np.unsignedinteger]: ...
    def k_core(self, *, degree: CoreDegree = "total", width: Width = "u8") -> npt.NDArray[np.unsignedinteger]: ...
    def clustering(self) -> DirectedClustering: ...
    def louvain(self) -> Louvain: ...
    def label_propagation(
        self,
        *,
        max_iterations: int = 40,
        tolerance: float = 0.05,
        exploration: int = 20,
        direction: Literal["both", "in", "out"] = "both",
    ) -> LabelPropagation: ...
    def betweenness(
        self,
        *,
        engine: Literal["auto", "exact", "kadabra"] = "auto",
        epsilon: float = 0.01,
        delta: float = 0.1,
        k: Optional[int] = None,
        normalized: bool = True,
        seed: int = 0x5EED,
    ) -> BetweennessResult: ...
    def spectral_clustering(
        self,
        k: Optional[int],
        *,
        matrix: Literal["laplacian", "bethe"] = "laplacian",
        engine: Literal["compressive", "exact"] = "compressive",
        regularization: Optional[float] = None,
        filter_degree: int = 64,
        moments: int = 128,
        probes: int = 16,
        features: Optional[int] = None,
        sample: Optional[int] = None,
        restarts: int = 4,
        refine: int = 3,
        seed: int = 0x5EED,
        max_k: int = 4096,
        subspace_degree: int = 16,
        tolerance: float = 1e-5,
        max_iterations: int = 500,
        precision: Literal["f32", "f64"] = "f32",
    ) -> Spectral: ...
    def spectral_embedding(
        self,
        count: int,
        *,
        matrix: Literal["laplacian", "bethe"] = "laplacian",
        regularization: Optional[float] = None,
        degree: int = 16,
        tolerance: float = 1e-5,
        max_iterations: int = 500,
        oversampling: Optional[int] = None,
        seed: int = 0x5EED,
        precision: Literal["f32", "f64"] = "f64",
    ) -> Embedding: ...
    def leiden(self) -> Louvain: ...
    def hyperball(
        self, *, precision: int = 8, normalized: bool = True, gpu: bool = False, direction: Direction = "out"
    ) -> HyperBall: ...
    def closeness(self, *, normalized: bool = True, direction: Direction = "out") -> npt.NDArray[np.float64]: ...
    def harmonic(self, *, normalized: bool = True, direction: Direction = "out") -> npt.NDArray[np.float64]: ...
    def lin(self, *, direction: Direction = "out") -> npt.NDArray[np.float64]: ...
    def pagerank(
        self,
        *,
        alpha: float = 0.85,
        tol: float = 1e-6,
        max_iter: int = 100,
        engine: Literal["auto", "partition"] = "auto",
    ) -> npt.NDArray[np.float64]: ...
    def degree_assortativity(
        self,
        *,
        x: Literal["out", "in"] = "out",
        y: Literal["out", "in"] = "in",
        method: Literal["pearson", "spearman"] = "pearson",
        gpu: bool = False,
    ) -> float: ...
    def numeric_assortativity(self, values: npt.ArrayLike) -> float: ...
    def attribute_assortativity(self, labels: npt.ArrayLike) -> float: ...
    def bfs(self, source: int, *, direction: Direction = "out") -> npt.NDArray[np.int64]: ...
    @overload
    def dijkstra(
        self, source: int, *, direction: Direction = "out", engine: Engine = "auto",
        delta: Optional[float] = None, predecessors: Literal[False] = False,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def dijkstra(
        self, source: int, *, direction: Direction = "out", engine: Engine = "auto",
        delta: Optional[float] = None, predecessors: Literal[True],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ...
    def build_landmarks(self, k: int = 8) -> npt.NDArray[np.int64]: ...
    @property
    def landmarks(self) -> Optional[npt.NDArray[np.int64]]: ...
    def drop_landmarks(self) -> None: ...
    def shortest_path(
        self,
        source: int,
        target: int,
        *,
        heuristic: Guide = "auto",
        bidirectional: Optional[bool] = None,
        paths: bool = True,
    ) -> Optional[Path]: ...
    def shortest_paths(
        self,
        sources: NodeIds,
        targets: NodeIds,
        *,
        heuristic: Guide = "auto",
        bidirectional: Optional[bool] = None,
        paths: bool = True,
    ) -> list[Optional[Path]]: ...

    def to_undirected(self, *, mode: UndirectedMode = "collapse", merge: WeightMerge = "sum") -> Graph:
        """The underlying undirected graph: "collapse" one edge per linked pair, "each" one per arc,
        "mutual" one per pair linked both ways; on a weighted graph, merged edges weigh their
        arcs' (both ways) "sum", "min" or "max". Owned by this graph: usable while it is open."""
