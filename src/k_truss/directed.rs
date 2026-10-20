//! Triangles of directed graphs.
//!
//! Seen from a directed edge `u -> v`, a third node `w` adjacent to both closes a triangle in one
//! of four (elementary) ways, the edge's *role* in it:
//!
//! * [`Cycle`] --- `v -> w` and `w -> u`: the edge is in a directed cycle `u -> v -> w -> u`.
//! * [`Out`] --- `u -> w` and `v -> w`: both endpoints point to `w`.
//! * [`In`] --- `w -> u` and `w -> v`: `w` points to both endpoints.
//! * [`Middleman`] --- `u -> w` and `w -> v`: `w` is a middleman between `u` and `v` (the edge is
//!   a shortcut of the path `u -> w -> v`).
//!
//! A feed-forward triangle `s -> m, m -> t, s -> t` holds one edge in each of the last three roles
//! (`s -> m` is [`Out`], `m -> t` is [`In`], `s -> t` is [`Middleman`]), while all of a cycle's edges
//! are [`Cycle`]. Reciprocal and parallel edges simply make more triangles: one per combination of
//! edges between the three nodes, as for undirected multigraphs.
//!
//! [`DirectedTriangleType`] selects any (non-empty) combination of roles: an edge's triangle count
//! is the number of triangles it is in, in one of the selected roles.
//!
//! [`Cycle`]: DirectedTriangleType::Cycle
//! [`Out`]: DirectedTriangleType::Out
//! [`In`]: DirectedTriangleType::In
//! [`Middleman`]: DirectedTriangleType::Middleman

/// Role bits, see the module's documentation.
pub(crate) const CYCLE: u8 = 1;
pub(crate) const OUT: u8 = 2;
pub(crate) const IN: u8 = 4;
pub(crate) const MIDDLEMAN: u8 = 8;

macro_rules! directed_triangle_types {
    ($($(#[$doc:meta])* $name:ident = $roles:expr, $prefix:literal;)*) => {
        /// The roles (see the [module's documentation](self)) in which an edge's triangles are
        /// counted: every elementary role and every combination of them.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum DirectedTriangleType {
            $($(#[$doc])* $name,)*
        }

        impl DirectedTriangleType {
            /// Every triangle type.
            pub const ALL: [DirectedTriangleType; 15] = [$(DirectedTriangleType::$name,)*];

            /// The selected roles' bits.
            pub(crate) const fn roles(self) -> u8 {
                match self {
                    $(DirectedTriangleType::$name => $roles,)*
                }
            }

            /// The type's name, also the prefix of its cached files' names.
            pub const fn name(self) -> &'static str {
                match self {
                    $(DirectedTriangleType::$name => $prefix,)*
                }
            }

            /// Name of the type's cached triangle counts.
            pub(crate) const fn triangles_suffix(self) -> &'static str {
                match self {
                    $(DirectedTriangleType::$name => concat!($prefix, "_triangles"),)*
                }
            }

            /// Name of the type's cached *PKT* k-truss decomposition.
            pub(crate) const fn pkt_suffix(self) -> &'static str {
                match self {
                    $(DirectedTriangleType::$name => concat!($prefix, "_ktrusspkt"),)*
                }
            }

            /// Name of the type's cached *Burkhardt et al.* k-truss decomposition.
            pub(crate) const fn bea_suffix(self) -> &'static str {
                match self {
                    $(DirectedTriangleType::$name => concat!($prefix, "_ktrussbea"),)*
                }
            }
        }
    };
}

directed_triangle_types! {
    /// Directed cycles.
    Cycle = CYCLE, "cycle";
    /// Edges whose endpoints point to a common node.
    Out = OUT, "out";
    /// Edges whose endpoints a common node points to.
    In = IN, "in";
    /// Edges with a middleman between their endpoints.
    Middleman = MIDDLEMAN, "middleman";
    CycleOut = CYCLE | OUT, "cycleout";
    CycleIn = CYCLE | IN, "cyclein";
    CycleMiddleman = CYCLE | MIDDLEMAN, "cyclemiddleman";
    OutIn = OUT | IN, "outin";
    OutMiddleman = OUT | MIDDLEMAN, "outmiddleman";
    InMiddleman = IN | MIDDLEMAN, "inmiddleman";
    CycleOutIn = CYCLE | OUT | IN, "cycleoutin";
    CycleOutMiddleman = CYCLE | OUT | MIDDLEMAN, "cycleoutmiddleman";
    CycleInMiddleman = CYCLE | IN | MIDDLEMAN, "cycleinmiddleman";
    /// Feed-forward triangles, in any role (`Out`, `In` or `Middleman`).
    FeedForward = OUT | IN | MIDDLEMAN, "feedforward";
    /// Every directed triangle, cycles and feed-forward ones.
    CycleFeedForward = CYCLE | OUT | IN | MIDDLEMAN, "cyclefeedforward";
}

use crate::graph::cache::AccumulatorAtomic;
use crate::graph::{self, CacheFile, DiGraph, IndexType};
use crate::shared_slice::{AbstractedProceduralMemoryMut, SharedSlice, SharedSliceMut};

use super::count::{TriangleCount, TriangleCounter};

use crossbeam::thread;
use portable_atomic::{AtomicBool, AtomicUsize, Ordering};
use std::mem::ManuallyDrop;
use std::path::Path;

impl DirectedTriangleType {
    /// Whether every triangle counts towards the support of all its edges or of none: unless
    /// only some feed-forward roles are selected. If so, an edge without support is in no
    /// triangle counting towards others' either.
    pub(crate) const fn symmetric(self) -> bool {
        let feed_forward = self.roles() & (OUT | IN | MIDDLEMAN);
        feed_forward == 0 || feed_forward == OUT | IN | MIDDLEMAN
    }
}

/// Nodes claimed by a thread at a time: dynamic scheduling balances uneven degrees.
const CHUNK: usize = 1024;

/// Runs `f` on every node `0..node_count`, on `threads` threads claiming chunks of nodes.
pub(crate) fn for_each_node(
    node_count: usize,
    threads: usize,
    f: impl Fn(usize) + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    let next = AtomicUsize::new(0);
    thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|_| {
                loop {
                    let begin = next.fetch_add(CHUNK, Ordering::Relaxed);
                    if begin >= node_count {
                        break;
                    }
                    for u in begin..std::cmp::min(begin + CHUNK, node_count) {
                        f(u);
                    }
                }
            });
        }
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })
}

/// The role (bit) of edge `a -> b` in a triangle with third node `c`, given the orientations of
/// its other two edges: whether the one between `a` and `c` is `a -> c`, and whether the one
/// between `b` and `c` is `b -> c`.
#[inline(always)]
pub(crate) const fn role(a_to_c: bool, b_to_c: bool) -> u8 {
    match (a_to_c, b_to_c) {
        (false, true) => CYCLE,
        (true, true) => OUT,
        (false, false) => IN,
        (true, false) => MIDDLEMAN,
    }
}

/// Entry of a node `u`'s merged adjacency: neighbour `node` (`w`), and the CSR offsets of the
/// first `u -> w` edge (in `u`'s neighbours) and of the first `w -> u` edge (in `w`'s), or
/// `<Ix as IndexType>::max()` if there's none. Parallel edges are adjacent in the CSR, so the rest follow.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Adjacent<Ix: IndexType> {
    pub(crate) node: Ix,
    out: Ix,
    inn: Ix,
}

/// A run of parallel edges: CSR offsets `start..start + len`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Run {
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl Run {
    #[inline(always)]
    pub(crate) fn edges(self) -> std::ops::Range<usize> {
        self.start..self.start + self.len
    }
}

/// Every node's in and out neighbours (other than itself), merged and sorted: the undirected
/// view of a directed graph, keeping track of each pair's edges in both directions.
pub(crate) struct DirectedAdjacency<'a, N: graph::N, E: graph::E, Ix: IndexType> {
    g: &'a DiGraph<N, E, Ix>,
    index_ptr: SharedSlice<Ix>,
    neighbours_ptr: SharedSlice<Ix>,
    offsets: AbstractedProceduralMemoryMut<Ix>,
    entries: AbstractedProceduralMemoryMut<Adjacent<Ix>>,
}

impl<'a, N: graph::N, E: graph::E, Ix: IndexType> DirectedAdjacency<'a, N, E, Ix> {
    /// Builds `g`'s merged adjacency, in temporary files (removed by [`cleanup`](Self::cleanup)).
    pub(crate) fn build(
        g: &'a DiGraph<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let edge_count = g.width();
        let index_ptr = SharedSlice::<Ix>::new(g.offsets_ptr(), g.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(g.neighbours_ptr(), edge_count);
        let file = |seq| g.build_cache_filename(CacheFile::DirectedAdjacency, Some(seq));

        // in-edges (but self-loops), as (source, edge offset) pairs grouped by destination: count,
        // prefix sum, scatter, then sort each destination's by source (and offset)
        // in-edges number at most the graph's edges, so their offsets fit `Ix`
        let cursors = SharedSliceMut::<Ix::Atomic>::abst_mem_mut(&file(0)?, node_count, true)?;
        let in_offsets = SharedSliceMut::<Ix>::abst_mem_mut(&file(1)?, node_count + 1, true)?;
        let in_edges = SharedSliceMut::<(Ix, Ix)>::abst_mem_mut(&file(2)?, edge_count, true)?;
        let counts = cursors.shared_slice();
        for_each_node(node_count, threads, |u| {
            for e in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                let v = neighbours_ptr.get(e).index();
                if v != u {
                    counts.get(v).increment(Ordering::Relaxed);
                }
            }
        })?;
        let in_ptr = {
            let mut in_offsets = in_offsets.shared_slice();
            let mut total = 0;
            for v in 0..node_count {
                *in_offsets.get_mut(v) = Ix::new(total);
                // cursors now track where each node's next in-edge goes
                let count = counts.get(v).load_widened(Ordering::Relaxed);
                counts.get(v).checked_store(total, Ordering::Relaxed)?;
                total += count;
            }
            *in_offsets.get_mut(node_count) = Ix::new(total);
            in_offsets
        };
        let scattered = in_edges.shared_slice();
        for_each_node(node_count, threads, |u| {
            let mut scattered = scattered;
            for e in index_ptr.get(u).index()..index_ptr.get(u + 1).index() {
                let v = neighbours_ptr.get(e).index();
                if v != u {
                    let at = counts.get(v).fetch_add_one_widened(Ordering::Relaxed);
                    *scattered.get_mut(at) = (Ix::new(u), Ix::new(e));
                }
            }
        })?;
        for_each_node(node_count, threads, |v| {
            let mut scattered = scattered;
            if let Some(list) =
                scattered.mut_slice(in_ptr.get(v).index(), in_ptr.get(v + 1).index())
            {
                list.sort_unstable_by_key(|&(u, e)| (u.index(), e.index()));
            }
        })?;

        // each node's out and in neighbours, merged: counted, then filled in
        let merge = |u: usize, emit: &mut dyn FnMut(Adjacent<Ix>)| {
            let (mut o, o_end) = (index_ptr.get(u).index(), index_ptr.get(u + 1).index());
            let (mut i, i_end) = (in_ptr.get(u).index(), in_ptr.get(u + 1).index());
            loop {
                while o < o_end && neighbours_ptr.get(o).index() == u {
                    o += 1;
                }
                let next_out = (o < o_end).then(|| neighbours_ptr.get(o).index());
                let next_in = (i < i_end).then(|| scattered.get(i).0.index());
                let w = match (next_out, next_in) {
                    (None, None) => break,
                    (Some(a), Some(b)) => a.min(b),
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                };
                let mut entry = Adjacent {
                    node: Ix::new(w),
                    out: <Ix as IndexType>::max(),
                    inn: <Ix as IndexType>::max(),
                };
                if next_out == Some(w) {
                    entry.out = Ix::new(o);
                    while o < o_end && neighbours_ptr.get(o).index() == w {
                        o += 1;
                    }
                }
                if next_in == Some(w) {
                    entry.inn = scattered.get(i).1;
                    while i < i_end && scattered.get(i).0.index() == w {
                        i += 1;
                    }
                }
                emit(entry);
            }
        };
        // a node's merged neighbours are at most its in and out edges, which fit `Ix`; all nodes'
        // can be up to twice the graph's edges, which is checked to fit
        let overflow = AtomicBool::new(false);
        for_each_node(node_count, threads, |u| {
            let mut count = 0;
            merge(u, &mut |_| count += 1);
            if counts
                .get(u)
                .checked_store(count, Ordering::Relaxed)
                .is_err()
            {
                overflow.store(true, Ordering::Relaxed);
            }
        })?;
        let offsets = SharedSliceMut::<Ix>::abst_mem_mut(&file(3)?, node_count + 1, true)?;
        let total = {
            let mut offsets = offsets.shared_slice();
            let mut total = 0;
            for u in 0..node_count {
                *offsets.get_mut(u) = Ix::new(total);
                total += counts.get(u).load_widened(Ordering::Relaxed);
            }
            *offsets.get_mut(node_count) = Ix::new(total);
            total
        };
        if overflow.load(Ordering::Relaxed) || total > <Ix as IndexType>::max().index() {
            g.remove_tmp_files(CacheFile::DirectedAdjacency, 0..5)?;
            return Err(format!(
                "error the graph's merged in and out neighbours ({total}) don't fit its index \
                 type ({}): use a wider one",
                std::any::type_name::<Ix>()
            )
            .into());
        }
        let entries = SharedSliceMut::<Adjacent<Ix>>::abst_mem_mut(&file(4)?, total, true)?;
        let (offsets_ptr, entries_ptr) = (offsets.shared_slice(), entries.shared_slice());
        for_each_node(node_count, threads, |u| {
            let mut entries = entries_ptr;
            let mut at = offsets_ptr.get(u).index();
            merge(u, &mut |entry| {
                *entries.get_mut(at) = entry;
                at += 1;
            });
        })?;

        Ok(Self {
            g,
            index_ptr,
            neighbours_ptr,
            offsets,
            entries,
        })
    }

    /// Removes the temporary files (and the in-triangle flags', see [`in_triangle_flags`]).
    ///
    /// [`in_triangle_flags`]: Self::in_triangle_flags
    pub(crate) fn cleanup(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.g.remove_tmp_files(CacheFile::DirectedAdjacency, 0..6)
    }

    /// A temporary file of per-edge flags (initially unset), e.g. for whether edges are in any
    /// triangle (removed by [`cleanup`](Self::cleanup)).
    pub(crate) fn in_triangle_flags(
        &self,
    ) -> Result<AbstractedProceduralMemoryMut<AtomicBool>, Box<dyn std::error::Error>> {
        let file = self
            .g
            .build_cache_filename(CacheFile::DirectedAdjacency, Some(5))?;
        SharedSliceMut::<AtomicBool>::abst_mem_mut(&file, self.g.width(), true)
    }

    /// Offsets of `u`'s entries.
    #[inline(always)]
    pub(crate) fn range(&self, u: usize) -> std::ops::Range<usize> {
        self.offsets.get(u).index()..self.offsets.get(u + 1).index()
    }

    #[inline(always)]
    pub(crate) fn entry(&self, i: usize) -> Adjacent<Ix> {
        *self.entries.get(i)
    }

    /// Offset of `u`'s first entry above `above`.
    #[inline(always)]
    pub(crate) fn first_above(&self, u: usize, above: usize) -> usize {
        let range = self.range(u);
        let entries = &self.entries.as_slice()[range.clone()];
        range.start + entries.partition_point(|a| a.node.index() <= above)
    }

    /// `u -> w` edges, for `u`'s entry of `w`.
    #[inline(always)]
    pub(crate) fn out_run(&self, u: usize, entry: Adjacent<Ix>) -> Run {
        Self::run(
            self,
            entry.out,
            entry.node.index(),
            self.index_ptr.get(u + 1).index(),
        )
    }

    /// `w -> u` edges, for `u`'s entry of `w`.
    #[inline(always)]
    pub(crate) fn in_run(&self, u: usize, entry: Adjacent<Ix>) -> Run {
        let w = entry.node.index();
        Self::run(self, entry.inn, u, self.index_ptr.get(w + 1).index())
    }

    /// The parallel edges to `target` from offset `start` (none if `start` is `<Ix as IndexType>::max()`) up to
    /// (at most) offset `end`.
    #[inline(always)]
    fn run(&self, start: Ix, target: usize, end: usize) -> Run {
        if start == <Ix as IndexType>::max() {
            return Run::default();
        }
        let start = start.index();
        let mut len = 1;
        while start + len < end && self.neighbours_ptr.get(start + len).index() == target {
            len += 1;
        }
        Run { start, len }
    }
}

/// Per-edge triangle counts of a directed graph, for a [`DirectedTriangleType`] (the roles in
/// which an edge's triangles count, see the [module's documentation](self)), as `Tx` integers
/// (`u8` by default, see [`TriangleCount`]).
///
/// [`TriangleCount`]: ../count/trait.TriangleCount.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct DirectedTriangles<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Tx: TriangleCount = u8,
> {
    g: &'a DiGraph<N, E, Ix>,
    triangle_type: DirectedTriangleType,
    /// Memmapped slice containing the number of triangles of each edge.
    triangles: AbstractedProceduralMemoryMut<Tx::Atomic>,
    threads: usize,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> DirectedTriangles<'a, N, E, Ix> {
    /// Counts every edge's triangles of type `triangle_type`, as `u8`s (see
    /// [`new_tx`](Self::new_tx) for other widths).
    pub fn new(
        g: &'a DiGraph<N, E, Ix>,
        triangle_type: DirectedTriangleType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_tx(g, triangle_type)
    }

    /// Searches for previously cached `u8` triangle counts of type `triangle_type`, computing
    /// them if not found.
    pub fn get_or_compute(
        g: &'a DiGraph<N, E, Ix>,
        triangle_type: DirectedTriangleType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::get_or_compute_tx(g, triangle_type)
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount>
    DirectedTriangles<'a, N, E, Ix, Tx>
{
    /// Counts every edge's triangles of type `triangle_type`, as `Tx` integers.
    pub fn new_tx(
        g: &'a DiGraph<N, E, Ix>,
        triangle_type: DirectedTriangleType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let adjacency = DirectedAdjacency::build(g, g.thread_num())?;
        let triangles = Self::new_with_adjacency(g, triangle_type, &adjacency, None);
        adjacency.cleanup()?;
        triangles
    }

    /// Counts every edge's triangles of type `triangle_type`, from `g`'s merged adjacency,
    /// flagging in `in_triangle` (if given) the edges in any (directed) triangle at all.
    pub(crate) fn new_with_adjacency(
        g: &'a DiGraph<N, E, Ix>,
        triangle_type: DirectedTriangleType,
        adjacency: &DirectedAdjacency<'a, N, E, Ix>,
        in_triangle: Option<SharedSliceMut<AtomicBool>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::DirectedTriangles(triangle_type), None)?;
        let triangles = SharedSliceMut::<Tx::Atomic>::abst_mem_mut(&out_fn, g.width(), true)?;
        let triangles = Self {
            g,
            triangle_type,
            triangles,
            threads: g.thread_num(),
        };
        triangles.compute(adjacency, in_triangle)?;
        Ok(triangles)
    }

    /// Searches for previously cached `Tx` triangle counts of type `triangle_type`, computing
    /// them if not found.
    pub fn get_or_compute_tx(
        g: &'a DiGraph<N, E, Ix>,
        triangle_type: DirectedTriangleType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t_fn = g.build_cache_filename(CacheFile::DirectedTriangles(triangle_type), None)?;
        if Path::new(&t_fn).exists() {
            // a cache of another graph (version) or count width would have another length
            if let Ok(triangles) =
                AbstractedProceduralMemoryMut::<Tx::Atomic>::from_file_name(&t_fn)
                && triangles.as_slice().len() == g.width()
            {
                return Ok(Self {
                    g,
                    triangle_type,
                    triangles,
                    threads: g.thread_num(),
                });
            }
        }
        Self::new_tx(g, triangle_type)
    }

    /// The type of the triangles counted.
    pub fn triangle_type(&self) -> DirectedTriangleType {
        self.triangle_type
    }

    pub fn triangles(&self, e_idx: usize) -> Tx {
        assert!(e_idx < self.g.width());
        Tx::from_usize(self.triangles.get(e_idx).count())
    }

    pub(crate) fn triangles_shares_slice(&self) -> SharedSliceMut<Tx::Atomic> {
        self.triangles.shared_slice()
    }

    pub fn all_triangles(&self) -> &[Tx::Atomic] {
        self.triangles.as_slice()
    }

    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this
            .g
            .build_cache_filename(CacheFile::DirectedTriangles(this.triangle_type), None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }

    /// Counts every edge's triangles. Every (undirected) triangle `x < y < z` of the merged
    /// adjacency is found exactly once, from its lowest pair `x, y`, by merging `x`'s and `y`'s
    /// neighbours above `y`. Each of its edges then counts, for each selected role it can play,
    /// the combinations of edges on the other two sides that put it in that role.
    fn compute(
        &self,
        adjacency: &DirectedAdjacency<'a, N, E, Ix>,
        in_triangle: Option<SharedSliceMut<AtomicBool>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let roles = self.triangle_type.roles();
        let tris = self.triangles.shared_slice();
        let overflow = AtomicBool::new(false);

        for_each_node(self.g.size(), self.threads, |x| {
            let x_range = adjacency.range(x);
            for i in x_range.clone() {
                let xy = adjacency.entry(i);
                let y = xy.node.index();
                if y <= x {
                    continue;
                }
                let (mut p, p_end) = (i + 1, x_range.end);
                let (mut q, q_end) = (adjacency.first_above(y, y), adjacency.range(y).end);
                // the `x, y` edges are the same for all of the pair's triangles
                let (x_to_y, y_to_x) = (adjacency.out_run(x, xy), adjacency.in_run(x, xy));
                while p < p_end && q < q_end {
                    let (xz, yz) = (adjacency.entry(p), adjacency.entry(q));
                    match xz.node.cmp(&yz.node) {
                        std::cmp::Ordering::Less => p += 1,
                        std::cmp::Ordering::Greater => q += 1,
                        std::cmp::Ordering::Equal => {
                            // runs[a][b]: the `a -> b` edges, for (a, b, c) = (x, y, z)
                            let z = xz.node.index();
                            let mut runs = [[Run::default(); 3]; 3];
                            runs[0][1] = x_to_y;
                            runs[1][0] = y_to_x;
                            runs[0][2] = adjacency.out_run(x, xz);
                            runs[2][0] = adjacency.in_run(x, xz);
                            runs[1][2] = adjacency.out_run(y, yz);
                            runs[2][1] = adjacency.in_run(y, yz);
                            debug_assert!(z > y);
                            if let Some(in_triangle) = in_triangle.as_ref() {
                                for run in runs.iter().flatten() {
                                    for e in run.edges() {
                                        in_triangle.get(e).store(true, Ordering::Relaxed);
                                    }
                                }
                            }
                            let m = |a: usize, b: usize| runs[a][b].len;
                            for (a, b, c) in [
                                (0, 1, 2),
                                (1, 0, 2),
                                (0, 2, 1),
                                (2, 0, 1),
                                (1, 2, 0),
                                (2, 1, 0),
                            ] {
                                if m(a, b) == 0 {
                                    continue;
                                }
                                let mut count = 0;
                                if roles & CYCLE != 0 {
                                    count += m(b, c) * m(c, a);
                                }
                                if roles & OUT != 0 {
                                    count += m(a, c) * m(b, c);
                                }
                                if roles & IN != 0 {
                                    count += m(c, a) * m(c, b);
                                }
                                if roles & MIDDLEMAN != 0 {
                                    count += m(a, c) * m(c, b);
                                }
                                if count == 0 {
                                    continue;
                                }
                                for e in runs[a][b].edges() {
                                    if tris.get(e).add_count(count) + count > Tx::MAX {
                                        overflow.store(true, Ordering::Relaxed);
                                    }
                                }
                            }
                            p += 1;
                            q += 1;
                        }
                    }
                }
            }
        })?;

        if overflow.load(Ordering::Relaxed) {
            return Err(format!(
                "error an edge is in more than {} triangles, more than can be counted ({})",
                Tx::MAX,
                std::any::type_name::<Tx>()
            )
            .into());
        }
        self.triangles.flush_async()?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::graph::test_utils::{unique_run_tag, unique_test_dir};
    use crate::k_truss::triangles::Triangles;
    use crate::k_truss::triangles::test::{
        DiGraph as TestDiGraph, DiTestGraph, fixtures, random_edges,
    };
    use std::collections::BTreeMap;
    use std::io::Write;

    /// Parses `edges` (0-based, directed) as a general MatrixMarket graph over `n` nodes.
    pub(crate) fn directed_graph(name: &str, n: usize, edges: &[(usize, usize)]) -> DiTestGraph {
        let dir = unique_test_dir("directed_triangles_test");
        std::fs::create_dir_all(&dir).expect("error creating tmp dir");
        let path = dir.join(format!("{name}.mtx"));
        let mut f = std::fs::File::create(&path).expect("error creating mtx fixture");
        writeln!(f, "%%MatrixMarket matrix coordinate pattern general").unwrap();
        writeln!(f, "{n} {n} {}", edges.len()).unwrap();
        for &(u, v) in edges {
            writeln!(f, "{} {}", u + 1, v + 1).unwrap();
        }
        drop(f);
        let graph = DiGraph::from_file(
            &path,
            Some(format!("dtri_{name}_{}", unique_run_tag())),
            Some(4),
        )
        .expect("graph parse should succeed");
        crate::graph::test_utils::TestGraph::new(graph, Some(dir))
    }

    /// A directed multigraph: random arcs, some reciprocated, some tripled, and self-loops.
    pub(crate) fn directed_multigraph_edges(n: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut edges = vec![];
        for (i, (u, v)) in random_edges(n, 0.15, seed).into_iter().enumerate() {
            // orient pseudo-randomly, reciprocate every 4th, triple every 7th
            let (u, v) = if (u * 31 + v * 17 + i) % 2 == 0 {
                (u, v)
            } else {
                (v, u)
            };
            edges.push((u, v));
            if i % 4 == 0 {
                edges.push((v, u));
            }
            if i % 7 == 0 {
                edges.extend([(u, v), (u, v)]);
            }
        }
        edges.extend([(0, 0), (3, 3), (3, 3)]);
        edges
    }

    /// Each CSR edge's triangles of the given roles by brute force (see the module's
    /// documentation): for edge `u -> v` and every other node `w`, the products of the
    /// multiplicities of the edges putting `u -> v` in each selected role.
    pub(crate) fn brute_force_directed(g: &TestDiGraph, roles: u8) -> Vec<usize> {
        let mut m = BTreeMap::<(usize, usize), usize>::new();
        for u in 0..g.size() {
            for v in g.neighbours(u).unwrap() {
                *m.entry((u, v)).or_insert(0) += 1;
            }
        }
        let m = |a: usize, b: usize| *m.get(&(a, b)).unwrap_or(&0);
        let mut counts = vec![];
        for u in 0..g.size() {
            for v in g.neighbours(u).unwrap() {
                let mut count = 0;
                if u != v {
                    for w in (0..g.size()).filter(|&w| w != u && w != v) {
                        if roles & CYCLE != 0 {
                            count += m(v, w) * m(w, u);
                        }
                        if roles & OUT != 0 {
                            count += m(u, w) * m(v, w);
                        }
                        if roles & IN != 0 {
                            count += m(w, u) * m(w, v);
                        }
                        if roles & MIDDLEMAN != 0 {
                            count += m(u, w) * m(w, v);
                        }
                    }
                }
                counts.push(count);
            }
        }
        counts
    }

    fn counts<Tx: TriangleCount>(
        t: &DirectedTriangles<'_, graph::label::VoidLabel, graph::label::VoidLabel, usize, Tx>,
    ) -> Vec<usize> {
        t.all_triangles().iter().map(|c| c.count()).collect()
    }

    #[test]
    fn directed_triangles_match_brute_force() {
        let g = directed_graph("multigraph", 60, &directed_multigraph_edges(60, 3));
        for triangle_type in DirectedTriangleType::ALL {
            let mut t = DirectedTriangles::<_, _, _, u16>::new_tx(&g, triangle_type).unwrap();
            assert_eq!(
                counts(&t),
                brute_force_directed(&g, triangle_type.roles()),
                "{triangle_type:?}"
            );
            t.drop_cache().unwrap();
        }
    }

    /// On a symmetric graph every role is played once per undirected triangle (combination of
    /// parallel edges), so each edge is in 4 times as many directed triangles as undirected ones.
    #[test]
    fn directed_triangles_of_symmetric_graphs() {
        for (name, g) in fixtures() {
            let undirected: Vec<usize> = Triangles::<_, _, _, u32>::new_tx(&g)
                .unwrap()
                .all_triangles()
                .iter()
                .map(|c| c.count())
                .collect();
            // the same graph, as a directed one
            let directed = crate::graph::DiGraph::<
                graph::label::VoidLabel,
                graph::label::VoidLabel,
                usize,
            >::open(g.cache_filename(), None, Some(4))
            .unwrap();
            let mut t = DirectedTriangles::<_, _, _, u32>::new_tx(
                &directed,
                DirectedTriangleType::CycleFeedForward,
            )
            .unwrap();
            let expected: Vec<usize> = undirected.iter().map(|c| 4 * c).collect();
            assert_eq!(counts(&t), expected, "{name}");
            t.drop_cache().unwrap();
        }
    }

    /// Every type caches its counts in its own file, `<name>_triangles`.
    #[test]
    fn directed_triangles_cache_per_type() {
        let g = directed_graph("cache", 30, &directed_multigraph_edges(30, 5));
        let file = |t: DirectedTriangleType| {
            g.build_cache_filename(CacheFile::DirectedTriangles(t), None)
                .unwrap()
        };
        let mut cycle = DirectedTriangles::new(&g, DirectedTriangleType::Cycle).unwrap();
        let mut feed_forward =
            DirectedTriangles::new(&g, DirectedTriangleType::FeedForward).unwrap();
        assert_ne!(
            file(DirectedTriangleType::Cycle),
            file(DirectedTriangleType::FeedForward)
        );
        assert!(file(DirectedTriangleType::Cycle).contains("cycle_triangles"));
        assert!(file(DirectedTriangleType::FeedForward).contains("feedforward_triangles"));
        assert!(Path::new(&file(DirectedTriangleType::Cycle)).exists());
        cycle.drop_cache().unwrap();
        assert!(!Path::new(&file(DirectedTriangleType::Cycle)).exists());
        assert!(Path::new(&file(DirectedTriangleType::FeedForward)).exists());
        // the cached counts are found again, and the temporary adjacency is gone
        let cached =
            DirectedTriangles::get_or_compute(&g, DirectedTriangleType::FeedForward).unwrap();
        assert_eq!(cached.all_triangles().len(), g.width());
        for seq in 0..5 {
            let tmp = g
                .build_cache_filename(CacheFile::DirectedAdjacency, Some(seq))
                .unwrap();
            assert!(!Path::new(&tmp).exists(), "{tmp} should have been removed");
        }
        feed_forward.drop_cache().unwrap();
    }
}
