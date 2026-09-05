"""Memory-mapped analytics for large static sparse graphs.

>>> import sparx
>>> with sparx.Graph("graph.mtx") as g:
...     coreness = g.k_core()
...     trussness = g.k_truss(algorithm="pkt")
"""

from ._sparx import DIRECTED_TRIANGLE_TYPES, GPU, Graph

__all__ = ["Graph", "DIRECTED_TRIANGLE_TYPES", "GPU"]
