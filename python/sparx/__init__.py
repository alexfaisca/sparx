"""Memory-mapped analytics for large static sparse graphs.

>>> import sparx
>>> with sparx.Graph("graph.mtx") as g:          # undirected
...     coreness = g.k_core()
...     trussness = g.k_truss(algorithm="pkt")
>>> with sparx.DiGraph("web.mtx") as d:          # directed
...     cores = d.k_core(degree="in")
...     cycles = d.k_truss("cycle")
"""

from ._sparx import DIRECTED_TRIANGLE_TYPES, GPU, DiGraph, Graph

__all__ = ["Graph", "DiGraph", "DIRECTED_TRIANGLE_TYPES", "GPU"]
