//! Visitor-style weighted searches, as [`BFSVisit`](super::bfs_visit::BFSVisit) and
//! [`DFSVisit`](super::dfs_visit::DFSVisit) are for traversals: a [`ShortestPathVisitor`]'s hooks
//! run as [`DijkstraVisit`] (Dijkstra's algorithm) or [`astar_visit`](super::astar::astar_visit)
//! (A*) search, and any of them can stop the search, by returning `ControlFlow::Break(())`.
//!
//! The hooks are Boost Graph Library's (`dijkstra_visitor`, `astar_visitor`), with distances:
//!
//! * [`discover_vertex`](ShortestPathVisitor::discover_vertex) --- a node is found, for the first
//!   time (the source first, as its own parent);
//! * [`examine_vertex`](ShortestPathVisitor::examine_vertex) --- a node is taken from the queue,
//!   to have its arcs relaxed: its distance is final (in A*, with a consistent heuristic;
//!   otherwise a node may be examined again, once found closer);
//! * [`examine_edge`](ShortestPathVisitor::examine_edge) --- each of its arcs, in turn;
//! * [`edge_relaxed`](ShortestPathVisitor::edge_relaxed) --- the arc shortens the path to its
//!   head (the head's discovery, the first time);
//! * [`edge_not_relaxed`](ShortestPathVisitor::edge_not_relaxed) --- it doesn't;
//! * [`finish_vertex`](ShortestPathVisitor::finish_vertex) --- all of a node's arcs were relaxed.
//!
//! Visitors' hooks are called through generics: a visitor that doesn't override a hook costs
//! nothing for it (the searches without a visitor run the same code, with a no-op one).

use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use super::dijkstra::{DijkstraMemory, dijkstra_visit};

use std::mem::ManuallyDrop;
use std::ops::ControlFlow;

/// Hooks of a weighted search (see the [module docs](self)), all no-ops by default: override
/// only what you need. Distances are `f64`s, arcs' weights too (`1` on unweighted graphs).
pub trait ShortestPathVisitor {
    /// `v` was found for the first time, at `distance`, through `parent` (the source is its own).
    #[inline(always)]
    fn discover_vertex(&mut self, _v: usize, _distance: f64, _parent: usize) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    /// `u`, at `distance`, is about to have its arcs relaxed.
    #[inline(always)]
    fn examine_vertex(&mut self, _u: usize, _distance: f64) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    /// The arc `u -> v`, of `weight`, is about to be relaxed.
    #[inline(always)]
    fn examine_edge(&mut self, _u: usize, _v: usize, _weight: f64) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    /// The arc `u -> v`, of `weight`, put `v` at `distance`, closer than it was found before.
    #[inline(always)]
    fn edge_relaxed(
        &mut self,
        _u: usize,
        _v: usize,
        _weight: f64,
        _distance: f64,
    ) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    /// The arc `u -> v`, of `weight`, didn't shorten the path to `v`.
    #[inline(always)]
    fn edge_not_relaxed(&mut self, _u: usize, _v: usize, _weight: f64) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    /// All of `u`'s arcs were relaxed.
    #[inline(always)]
    fn finish_vertex(&mut self, _u: usize, _distance: f64) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// A visitor doing nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoOpShortestPathVisitor;
impl ShortestPathVisitor for NoOpShortestPathVisitor {}

/// Two visitors, run one after the other at each hook (the second not at all, if the first
/// stops the search).
pub struct ShortestPathBundle<A, B>(pub A, pub B);

macro_rules! both {
    ($self:ident.$hook:ident($($arg:expr),*)) => {
        match $self.0.$hook($($arg),*) {
            ControlFlow::Break(()) => ControlFlow::Break(()),
            ControlFlow::Continue(()) => $self.1.$hook($($arg),*),
        }
    };
}

impl<A: ShortestPathVisitor, B: ShortestPathVisitor> ShortestPathVisitor
    for ShortestPathBundle<A, B>
{
    fn discover_vertex(&mut self, v: usize, d: f64, p: usize) -> ControlFlow<()> {
        both!(self.discover_vertex(v, d, p))
    }
    fn examine_vertex(&mut self, u: usize, d: f64) -> ControlFlow<()> {
        both!(self.examine_vertex(u, d))
    }
    fn examine_edge(&mut self, u: usize, v: usize, w: f64) -> ControlFlow<()> {
        both!(self.examine_edge(u, v, w))
    }
    fn edge_relaxed(&mut self, u: usize, v: usize, w: f64, d: f64) -> ControlFlow<()> {
        both!(self.edge_relaxed(u, v, w, d))
    }
    fn edge_not_relaxed(&mut self, u: usize, v: usize, w: f64) -> ControlFlow<()> {
        both!(self.edge_not_relaxed(u, v, w))
    }
    fn finish_vertex(&mut self, u: usize, d: f64) -> ControlFlow<()> {
        both!(self.finish_vertex(u, d))
    }
}

/// Dijkstra's algorithm from a `source` with a [`ShortestPathVisitor`]: sequential (on one
/// thread, as a visitor is called in order), with
/// [`DijkstraDists`](super::dijkstra::DijkstraDists)' memory (`O(|V|)`, mapped in the cache).
///
/// It follows arcs out of nodes, on directed graphs: pass it a [`view`] of the graph to follow
/// them in another [`Direction`].
///
/// [`view`]: ../../graph/struct.GraphMemoryMap.html#method.view
/// [`Direction`]: ../../graph/direction/enum.Direction.html
pub struct DijkstraVisit<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness> {
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    /// Each node's distance: final for examined nodes, tentative for nodes only discovered (if
    /// the visitor stopped the search), infinite for the rest.
    distances: AbstractedProceduralMemoryMut<f64>,
    stopped: bool,
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Dir: Directedness>
    DijkstraVisit<'a, N, E, Ix, Dir>
{
    /// Runs Dijkstra's algorithm from `source`, calling `visitor`'s hooks, until every node
    /// `source` reaches is examined or the visitor stops it.
    pub fn new<V: ShortestPathVisitor + ?Sized>(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        source: usize,
        visitor: &mut V,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n = g.size();
        if source >= n {
            return Err(
                format!("error node with id {source} doesn't exist: max id is |V| = {n}").into(),
            );
        }
        // checks the weights finite and non-negative
        g.total_weight()?;
        let out_fn = g.build_cache_filename(CacheFile::Dijkstra, Some(10))?;
        let scratch: Vec<String> = (11..13)
            .map(|k| g.build_cache_filename(CacheFile::Dijkstra, Some(k)))
            .collect::<Result<_, _>>()?;
        let result = (|| -> Result<_, Box<dyn std::error::Error>> {
            let mut m = DijkstraMemory::<Ix>::new(&scratch, n)?;
            let (_, stopped) =
                dijkstra_visit(g, source, &mut m, visitor).map_err(|e| e.to_string())?;
            let mut distances = SharedSliceMut::<f64>::abst_mem_mut(&out_fn, n, true)?;
            for (v, d) in distances.as_mut_slice().iter_mut().enumerate() {
                *d = m.distance(v).unwrap_or(f64::INFINITY);
            }
            distances.flush()?;
            Ok((distances, stopped))
        })();
        for f in &scratch {
            let _ = std::fs::remove_file(f);
        }
        let (distances, stopped) = result?;
        Ok(Self {
            g,
            distances,
            stopped,
        })
    }

    /// Each node's distance from the source: final for the nodes the search examined, tentative
    /// for those it only discovered (if the visitor stopped it), [`f64::INFINITY`] for the rest.
    pub fn get_distances(&self) -> &[f64] {
        self.distances.as_slice()
    }

    /// Whether the visitor stopped the search.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Removes the distances from the cache.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::Dijkstra, Some(10))?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}
