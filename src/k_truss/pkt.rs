use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::{AtomicUsize, Ordering};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::{Arc, Barrier};

use super::count::{TriangleCount, TriangleCounter};
use super::directed::{DirectedAdjacency, DirectedTriangleType, DirectedTriangles};
use super::triangles::Triangles;
use super::{TriangleIndex, check_support_range};

/// Edges of a sub-level a thread claims at a time.
const POP_CHUNK: usize = 64;

type ProceduralMemoryPKT<Ix> = (
    AbstractedProceduralMemoryMut<Ix>,
    AbstractedProceduralMemoryMut<Ix>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<bool>,
);

/// For the computation of a [`GraphMemoryMap`] instance's k-truss decomposition as described in ["Shared-memory Graph Truss Decomposition"](https://doi.org/10.48550/arXiv.1707.02000) by Kamir H. and Madduri K.
///
/// Trussness is stored as `Tx` integers (`u8` by default, see [`TriangleCount`]). For directed
/// graphs (see [`new_directed`](Self::new_directed)), the decomposition is over the triangles
/// of a [`DirectedTriangleType`].
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
/// [`TriangleCount`]: ../count/trait.TriangleCount.html#
/// [`DirectedTriangleType`]: ../directed/enum.DirectedTriangleType.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoPKT<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount = u8> {
    /// Graph for which edge trussness is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing the trussness of each edge.
    k_trusses: AbstractedProceduralMemoryMut<Tx>,
    threads: usize,
    /// The triangles' type, for directed graphs.
    directed: Option<DirectedTriangleType>,
}

/// Cached file of a decomposition (of an undirected graph, or of a directed one's triangles of a
/// given type).
fn cache_file(directed: Option<DirectedTriangleType>) -> CacheFile {
    match directed {
        None => CacheFile::KTrussPKT,
        Some(t) => CacheFile::KTrussPKTDirected(t),
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoPKT<'a, N, E, Ix> {
    /// Performs the *PKT Alrgorithm*'s k-truss decomposition as described in ["Shared-memory Graph Truss Decomposition"](https://doi.org/10.48550/arXiv.1707.02000) by Kamir H. and Madduri K.
    ///
    /// Trussness is stored as `u8`s, see [`new_tx`](Self::new_tx) for other widths.
    ///
    /// # Arguments
    ///
    /// * `g` --- the  [`GraphMemoryMap`] instance in which k-truss decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_tx(g, g.thread_num())
    }

    /// Searches for a previously cached result, and if not found performs the *PKT Alrgorithm*'s k-truss decomposition as described in ["Shared-memory Graph Truss Decomposition"](https://doi.org/10.48550/arXiv.1707.02000) by Kamir H. and Madduri K.
    ///
    /// # Arguments
    ///
    /// * `g` --- the  [`GraphMemoryMap`] instance in which k-truss decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn get_or_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::get_or_compute_tx(g)
    }

    /// Performs the *PKT Alrgorithm*'s 'k-truss decomposition as described in ["Shared-memory Graph Truss Decomposition"](https://doi.org/10.48550/arXiv.1707.02000) by Kamir H. and Madduri K.
    ///
    /// # Arguments
    ///
    /// * `g` --- the  [`GraphMemoryMap`] instance in which k-truss decomposition is to be performed in.
    /// * `threads` --- the number of threads to be used in the computation.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new_with_conf(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_tx(g, threads)
    }

    /// Performs the *PKT Alrgorithm*'s k-truss decomposition of a directed graph, over its
    /// triangles of type `triangle_type` (an edge is in the k-truss if it's in at least `k - 2`
    /// triangles, in a selected role, of the k-truss). Trussness is stored as `u8`s, see
    /// [`new_directed_tx`](Self::new_directed_tx) for other widths.
    pub fn new_directed(
        g: &'a GraphMemoryMap<N, E, Ix>,
        triangle_type: DirectedTriangleType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_directed_tx(g, triangle_type, g.thread_num())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount>
    AlgoPKT<'a, N, E, Ix, Tx>
{
    /// Performs the *PKT Alrgorithm*'s k-truss decomposition, storing trussness as `Tx` integers.
    pub fn new_tx(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut pkt = Self::new_no_compute(g, threads)?;
        let proc_mem = pkt.init_cache_mem()?;

        pkt.compute_with_proc_mem(proc_mem)?;

        Ok(pkt)
    }

    /// Searches for a previously cached `Tx` result, computing it if not found.
    pub fn get_or_compute_tx(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t_fn = g.build_cache_filename(CacheFile::KTrussPKT, None)?;
        if Path::new(&t_fn).exists() {
            // a cache of another graph (version) or trussness width would have another length
            if let Ok(k_trusses) = AbstractedProceduralMemoryMut::<Tx>::from_file_name(&t_fn)
                && k_trusses.as_slice().len() == g.width()
            {
                return Ok(Self {
                    g,
                    k_trusses,
                    threads: g.thread_num(),
                    directed: None,
                });
            }
        }
        Self::new_tx(g, g.thread_num())
    }

    /// Performs the *PKT Alrgorithm*'s k-truss decomposition from precomputed `triangles` of `g`
    /// (e.g. counted on a GPU, with [`Triangles::new_gpu`]), which are consumed: their counts are
    /// used as working memory.
    ///
    /// [`Triangles::new_gpu`]: ../triangles/struct.Triangles.html#method.new_gpu
    pub fn new_with_triangles(
        g: &'a GraphMemoryMap<N, E, Ix>,
        triangles: Triangles<'a, N, E, Ix, Tx>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let pkt = Self::new_no_compute(g, g.thread_num())?;
        let proc_mem = pkt.init_cache_mem()?;
        pkt.compute_with_triangles(proc_mem, triangles)?;
        Ok(pkt)
    }

    /// Performs the *PKT Alrgorithm*'s k-truss decomposition of a directed graph, over its
    /// triangles of type `triangle_type`, storing trussness as `Tx` integers.
    pub fn new_directed_tx(
        g: &'a GraphMemoryMap<N, E, Ix>,
        triangle_type: DirectedTriangleType,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // the merged adjacency serves both counting and peeling
        let adjacency = DirectedAdjacency::build(g, threads)?;
        let result = (|| {
            // with only some roles selected, edges without support are peeled too, but only those
            // in some triangle can matter
            let in_triangle = match triangle_type.symmetric() {
                true => None,
                false => Some(adjacency.in_triangle_flags()?),
            };
            let triangles = DirectedTriangles::<N, E, Ix, Tx>::new_with_adjacency(
                g,
                triangle_type,
                &adjacency,
                in_triangle.as_ref().map(|f| f.shared_slice()),
            )?;
            let pkt = Self::new_no_compute_impl(g, threads, Some(triangle_type))?;
            let proc_mem = pkt.init_cache_mem()?;
            pkt.compute_directed_with_triangles(
                proc_mem,
                triangles,
                Some((&adjacency, in_triangle.as_ref().map(|f| f.shared_slice()))),
            )?;
            Ok(pkt)
        })();
        adjacency.cleanup()?;
        result
    }

    /// Performs the *PKT Alrgorithm*'s k-truss decomposition of a directed graph from its
    /// precomputed `triangles` (of their type), which are consumed: their counts are used as
    /// working memory.
    pub fn new_directed_with_triangles(
        g: &'a GraphMemoryMap<N, E, Ix>,
        triangles: DirectedTriangles<'a, N, E, Ix, Tx>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let pkt = Self::new_no_compute_impl(g, g.thread_num(), Some(triangles.triangle_type()))?;
        let proc_mem = pkt.init_cache_mem()?;
        pkt.compute_directed_with_triangles(proc_mem, triangles, None)?;
        Ok(pkt)
    }

    /// Returns the trussness of a given edge of a [`GraphMemoryMap`] instance.
    ///
    /// # Arguments
    ///
    /// * `e_idx` --- the index of the edge whose trussness is to be returned.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn trussness(&self, e_idx: usize) -> Tx {
        assert!(e_idx < self.g.width());
        *self.k_trusses.get(e_idx)
    }

    /// Returns a slice containing the trussness of each edge of the [`GraphMemoryMap`] instance.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn k_trusses(&self) -> &[Tx] {
        self.k_trusses.as_slice()
    }

    /// The triangles' type, for decompositions of directed graphs.
    pub fn triangle_type(&self) -> Option<DirectedTriangleType> {
        self.directed
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this
            .g
            .build_cache_filename(cache_file(this.directed), None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount>
    AlgoPKT<'a, N, E, Ix, Tx>
{
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, threads, None)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, threads, None)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(&self) -> Result<ProceduralMemoryPKT<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryPKT<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryPKT<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryPKT<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.width()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
        threads: usize,
        directed: Option<DirectedTriangleType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if directed.is_none() {
            g.require_symmetric("PKT", Some("`AlgoPKT::new_directed`"))?;
        }
        let threads = threads.max(1);
        let out_fn = g.build_cache_filename(cache_file(directed), None)?;
        let k_trusses = SharedSliceMut::<Tx>::abst_mem_mut(&out_fn, g.width(), true)?;
        Ok(Self {
            g,
            k_trusses,
            threads,
            directed,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<ProceduralMemoryPKT<Ix>, Box<dyn std::error::Error>> {
        let edge_count = self.g.width();
        let file = |seq| {
            self.g
                .build_cache_filename(cache_file(self.directed), Some(seq))
        };

        let curr = SharedSliceMut::<Ix>::abst_mem_mut(&file(0)?, edge_count, true)?;
        let next = SharedSliceMut::<Ix>::abst_mem_mut(&file(1)?, edge_count, true)?;
        let processed = SharedSliceMut::<bool>::abst_mem_mut(&file(2)?, edge_count, true)?;
        let in_curr = SharedSliceMut::<bool>::abst_mem_mut(&file(3)?, edge_count, true)?;
        let in_next = SharedSliceMut::<bool>::abst_mem_mut(&file(4)?, edge_count, true)?;

        // pre-initialize the memmapped files if they don't exist
        if self.directed.is_none() {
            let _edge_reciprocal = self.g.edge_reciprocal()?;
            let _edge_out = self.g.edge_over()?;
        }

        Ok((curr, next, processed, in_curr, in_next))
    }

    fn compute_with_proc_mem_impl(
        &self,
        proc_mem: ProceduralMemoryPKT<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_triangles(proc_mem, Triangles::new_tx(self.g)?)
    }

    /// Decomposes the (undirected) graph from its `triangles`, whose counts are used (and
    /// consumed) as working memory.
    fn compute_with_triangles(
        &self,
        proc_mem: ProceduralMemoryPKT<Ix>,
        mut triangles: Triangles<'a, N, E, Ix, Tx>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if triangles.all_triangles().len() != self.g.width() {
            return Err(format!(
                "error expected the triangles of a graph with {} edges, got {}",
                self.g.width(),
                triangles.all_triangles().len()
            )
            .into());
        }
        let edge_reciprocal = self.g.edge_reciprocal()?;
        let index = TriangleIndex::Undirected {
            offsets: SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size()),
            neighbours: SharedSlice::<Ix>::new(self.g.neighbours_ptr(), self.g.width()),
            reciprocal: edge_reciprocal.shared_slice(),
        };
        self.peel(proc_mem, triangles.triangles_shares_slice(), &index)?;
        triangles.drop_cache()?;
        Ok(())
    }

    /// Decomposes the directed graph from its `triangles`, whose counts are used (and consumed)
    /// as working memory.
    fn compute_directed_with_triangles(
        &self,
        proc_mem: ProceduralMemoryPKT<Ix>,
        mut triangles: DirectedTriangles<'a, N, E, Ix, Tx>,
        adjacency: Option<(
            &DirectedAdjacency<'a, N, E, Ix>,
            Option<SharedSliceMut<portable_atomic::AtomicBool>>,
        )>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if triangles.all_triangles().len() != self.g.width() {
            return Err(format!(
                "error expected the triangles of a graph with {} edges, got {}",
                self.g.width(),
                triangles.all_triangles().len()
            )
            .into());
        }
        // the caller's merged adjacency, or one built (and removed) here
        let built = match adjacency {
            Some(_) => None,
            None => Some(DirectedAdjacency::build(self.g, self.threads)?),
        };
        let (adjacency, in_triangle) = match (adjacency, built.as_ref()) {
            (Some((adjacency, in_triangle)), _) => (adjacency, in_triangle),
            (None, Some(built)) => (built, None),
            (None, None) => return Err("error no merged adjacency".into()),
        };
        let index = TriangleIndex::Directed {
            adjacency,
            offsets: SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size()),
            roles: triangles.triangle_type().roles(),
            in_triangle,
        };
        let result = self.peel(proc_mem, triangles.triangles_shares_slice(), &index);
        if let Some(built) = built {
            built.cleanup()?;
        }
        result?;
        triangles.drop_cache()?;
        Ok(())
    }

    /// ParTriangle-AM4 --- adjusted for multigraphs and directed triangle types: peels edges by
    /// increasing support `s` (their triangle counts, consumed), storing their trussness.
    fn peel(
        &self,
        proc_mem: ProceduralMemoryPKT<Ix>,
        s: SharedSliceMut<Tx::Atomic>,
        index: &TriangleIndex<'_, 'a, N, E, Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let edge_count = self.g.width();

        let threads = self.threads;
        let edge_load = edge_count.div_ceil(threads);

        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);

        // Shared arrays
        let (curr, next, processed, in_curr, in_next) = proc_mem;

        // Thread syncronization
        let synchronize = Arc::new(Barrier::new(threads));

        thread::scope(|scope| {
            for tid in 0..threads {
                // eid is unnecessary as graph + index alwready do the job
                let mut curr = curr.shared_slice();
                let mut next = next.shared_slice();
                let mut in_curr = in_curr.shared_slice();
                let mut in_next = in_next.shared_slice();
                let mut processed = processed.shared_slice();

                let begin = std::cmp::min(tid * edge_load, edge_count);
                let end = std::cmp::min(begin + edge_load, edge_count);
                scope.spawn(move |_| {
                    // initialize edge_out, x, curr, next, in_curr, in_next & processed
                    for edge_offset in begin..end {
                        *curr.get_mut(edge_offset) = Ix::new(0);
                        *next.get_mut(edge_offset) = Ix::new(0);
                        *in_curr.get_mut(edge_offset) = false;
                        *in_next.get_mut(edge_offset) = false;
                        *processed.get_mut(edge_offset) = false;
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        check_support_range::<Tx>(s.as_slice())?;
        // with only some roles selected, an edge without triangles counting towards its support
        // may still be in triangles counting towards others': it's peeled (at level 0) like any
        // other edge, rather than just discarded
        let peel_unsupported = !index.symmetric();
        let mut l: usize = if peel_unsupported { 0 } else { 1 };
        let buff_size = 4096;
        let total_duds = Arc::new(AtomicUsize::new(0));
        let curr = SharedQueueMut::<Ix>::from_shared_slice(curr.shared_slice());
        let next = SharedQueueMut::<Ix>::from_shared_slice(next.shared_slice());

        thread::scope(|scope| {
            let mut res = Vec::new();
            for tid in 0..threads {
                let mut todo = edge_count;

                let s = s.clone();
                let mut curr = curr.clone();
                let mut next = next.clone();
                let mut in_curr = in_curr.shared_slice();
                let mut in_next = in_next.shared_slice();
                let mut processed = processed.shared_slice();
                let mut trussness = self.k_trusses.shared_slice();

                let total_duds = Arc::clone(&total_duds);
                let synchronize = Arc::clone(&synchronize);

                let begin = std::cmp::min(tid * edge_load, edge_count);
                let end = std::cmp::min(begin + edge_load, edge_count);

                res.push(scope.spawn(
                    move |_| -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync>> {
                        // edges per support level (for verbose output)
                        let mut res = vec![0usize; 1];
                        let mut buff = vec![Ix::new(0); buff_size];
                        let mut i = 0;

                        // Remove 0-triangle edges
                        for e in begin..end {
                            if s.get(e).count() == 0
                                && (!peel_unsupported || !index.may_be_in_triangle(e))
                            {
                                *processed.get_mut(e) = true;
                                res[0] += 1;
                                i += 1;
                            }
                        }
                        total_duds.fetch_add(i, Ordering::SeqCst);
                        i = 0;

                        synchronize.wait();

                        todo = match todo.overflowing_sub(total_duds.load(Ordering::Relaxed)) {
                            (r, false) => r,
                            _ => {
                                return Err(format!(
                                    "error overflow when decrementing todo ({todo} - {})",
                                    total_duds.load(Ordering::Relaxed)
                                )
                                .into());
                            }
                        };

                        // decrements edge `$x`'s support (above `l`), moving it to the next
                        // sub-level if it reaches `l`
                        macro_rules! decrement {
                            ($x:expr) => {{
                                let x = $x;
                                let prev = s.get(x).sub_count(1);
                                if prev == l + 1 {
                                    *in_next.get_mut(x) = true;
                                    buff[i] = Ix::new(x);
                                    i += 1;
                                    if i == buff_size {
                                        next.push_slice(&buff[..]);
                                        i = 0;
                                    }
                                }
                                if prev <= l {
                                    s.get(x).set_count(l);
                                }
                            }};
                        }

                        let mut closing = Vec::new();
                        while todo > 0 {
                            for e in begin..end {
                                // (edges without support set aside up front are already processed)
                                if s.get(e).count() == l && !*processed.get(e) {
                                    buff[i] = Ix::new(e);
                                    *in_curr.get_mut(e) = true;
                                    i += 1;
                                }
                                if i == buff_size {
                                    curr.push_slice(buff.as_slice());
                                    i = 0;
                                }
                            }
                            if i > 0 {
                                curr.push_slice(&buff[0..i]);
                                i = 0;
                            }
                            synchronize.wait();

                            while !curr.is_empty() {
                                todo = match todo.overflowing_sub(curr.len()) {
                                    (r, false) => r,
                                    _ => {
                                        return Err(format!(
                                            "error overflow when decrementing todo ({todo} - {})",
                                            curr.len()
                                        )
                                        .into());
                                    }
                                };
                                let to_process = curr.slice(0, curr.len()).ok_or_else(
                                    || -> Box<dyn std::error::Error + Send + Sync> {
                                        "error reading curr in pkt".into()
                                    },
                                )?;
                                synchronize.wait();

                                // ProcessSubLevel
                                let thread_load = curr.len().div_ceil(threads);
                                let begin = std::cmp::min(tid * thread_load, curr.len());
                                let end = std::cmp::min(begin + thread_load, curr.len());

                                // edges are claimed in chunks: one contended update per chunk
                                while let Some(chunk) = curr.pop_chunk(POP_CHUNK) {
                                    for u_v in chunk {
                                        let u_v = u_v.index();
                                        let u = index.source(u_v);
                                        let v = neighbours_ptr.get(u_v).index();

                                        index.closing(u, v, &mut closing);
                                        for c in closing.iter() {
                                            let (f, g) = (c.f, c.g);
                                            if *processed.get(f) || *processed.get(g) {
                                                continue;
                                            }
                                            // the triangle's support is taken from each of its other
                                            // edges (whose role in it counts) still above `l` once:
                                            // by `u -> v` unless the third edge is also in this
                                            // sub-level, and lower (it then takes it itself)
                                            if c.f_counts
                                                && s.get(f).count() > l
                                                && (u_v < g || !*in_curr.get(g))
                                            {
                                                decrement!(f);
                                            }
                                            if c.g_counts
                                                && s.get(g).count() > l
                                                && (u_v < f || !*in_curr.get(f))
                                            {
                                                decrement!(g);
                                            }
                                        }
                                    }
                                }
                                if i > 0 {
                                    next.push_slice(&buff[0..i]);
                                    i = 0;
                                }

                                // the sub-level's edges are only marked processed once every thread
                                // is done with them: until then, triangles between them must still be
                                // seen (and counted once, by the `in_curr` tie-break above)
                                synchronize.wait();

                                if res.len() <= l {
                                    res.resize(l + 1, 0);
                                }
                                for e_idx in begin..end {
                                    let edge = to_process.get(e_idx).index();
                                    *processed.get_mut(edge) = true;
                                    *in_curr.get_mut(edge) = false;
                                    res[l] += 1;
                                }

                                synchronize.wait();

                                if tid == 0 {
                                    curr.clone().clear();
                                }
                                next = std::mem::replace(&mut curr, next);
                                in_next = std::mem::replace(&mut in_curr, in_next);

                                synchronize.wait();
                                synchronize.wait();
                            }
                            l += 1;
                            synchronize.wait();
                        }

                        synchronize.wait();

                        // by definition any non-trivial subgraph is at least a 2-truss: other
                        // values go up from there (supports were checked to fit `Tx - 2`).
                        for e in begin..end {
                            *trussness.get_mut(e) = Tx::from_usize(s.get(e).count() + 2);
                        }
                        Ok(res)
                    },
                ));
            }
            let joined_res: Vec<Vec<usize>> = res
                .into_iter()
                .map(|v| v.join().expect("error thread panicked").expect("error"))
                .collect();
            let env_verbose_val =
                std::env::var("BRUIJNX_VERBOSE").unwrap_or_else(|_| "0".to_string());
            let verbose: bool = env_verbose_val == "1";
            if verbose {
                // edges per trussness (support + 2)
                let levels = joined_res.iter().map(|v| v.len()).max().unwrap_or(0);
                let mut r = vec![0usize; levels + 2];
                for v in joined_res.iter() {
                    for (support, count) in v.iter().enumerate() {
                        r[support + 2] += count;
                    }
                }
                println!("k-trussness {:?}", r);
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        // cleanup cache (curr, next, processed, in_curr and in_next)
        self.g.remove_tmp_files(cache_file(self.directed), 0..5)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::k_truss::verify_k_trusses;

    use super::*;
    use paste::paste;
    use std::path::Path;

    macro_rules! graph_tests {
        ($($name:ident => $path:expr ,)*) => {
            $(
                paste! {
                    #[test]
                    fn [<k_trusses_pkt_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
                        generic_test($path)
                    }
                }
            )*
        }
    }

    fn generic_test<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn std::error::Error>> {
        // a private copy: tests on the same fixture may run concurrently
        let (graph, _cache) = crate::graph::test_utils::open_fixture(path.as_ref(), Some(16))?;
        let pkt_k_trusses = AlgoPKT::new(&graph)?;

        verify_k_trusses(&graph, pkt_k_trusses.k_trusses())
    }

    // generate test cases from dataset
    graph_tests! {
        ggcat_1_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_1_5.lz4"),
        ggcat_2_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_2_5.lz4"),
        ggcat_3_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_3_5.lz4"),
        ggcat_4_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_4_5.lz4"),
        ggcat_5_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_5_5.lz4"),
        ggcat_6_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_6_5.lz4"),
        ggcat_7_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_7_5.lz4"),
        ggcat_8_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_8_5.lz4"),
        // ggcat_9_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_9_5.lz4"),
        // ggcat_1_10 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_1_10.lz4"),
        // ggcat_2_10 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_2_10.lz4"),
        // ggcat_3_10 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_3_10.lz4"),
        // ggcat_4_10 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_4_10.lz4"),
        // ggcat_5_10 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_5_10.lz4"),
        // ggcat_6_10 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_6_10.lz4"),
        // ggcat_7_10 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_7_10.lz4"),
        // ggcat_8_10 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_8_10.lz4"),
        // ggcat_9_10 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_9_10.lz4"),
        // ggcat_8_15 => "../ggcat/graphs/random_graph_8_15.lz4",
        // ggcat_9_15 => "../ggcat/graphs/random_graph_9_15.lz4",
        // … add the rest
    }
}
