use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use num_cpus::get_physical;
use std::mem::ManuallyDrop;
use std::path::Path;

use super::count::{TriangleCount, TriangleCounter};
use super::directed::{DirectedAdjacency, DirectedTriangleType, DirectedTriangles};
use super::triangles::Triangles;
use super::{TriangleIndex, check_support_range};

type ProceduralMemoryBurkhardtEtAl<Ix> = (
    AbstractedProceduralMemoryMut<Ix>,
    AbstractedProceduralMemoryMut<Ix>,
    AbstractedProceduralMemoryMut<(Ix, Ix)>,
);

/// For the computation of a [`GraphMemoryMap`] instance's k-truss decomposition as described in ["Bounds and algorithms for graph trusses"](https://doi.org/10.48550/arXiv.1806.05523) by Burkhardt P. et al.
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
pub struct AlgoBurkhardtEtAl<
    'a,
    N: graph::N,
    E: graph::E,
    Ix: graph::IndexType,
    Tx: TriangleCount = u8,
    Dir: Directedness = Undirected,
> {
    /// Graph for which edge trussness is computed.
    g: &'a GraphMemoryMap<N, E, Ix, Dir>,
    /// Memmapped slice containing the trussness of each edge.
    k_trusses: AbstractedProceduralMemoryMut<Tx>,
    /// The triangles' type, for directed graphs.
    directed: Option<DirectedTriangleType>,
}

/// Cached file of a decomposition (of an undirected graph, or of a directed one's triangles of a
/// given type).
fn cache_file(directed: Option<DirectedTriangleType>) -> CacheFile {
    match directed {
        None => CacheFile::KTrussBEA,
        Some(t) => CacheFile::KTrussBEADirected(t),
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoBurkhardtEtAl<'a, N, E, Ix> {
    /// Performs k-truss decomposition as described in ["Bounds and algorithms for graph trusses"](https://doi.org/10.48550/arXiv.1806.05523) by Burkhardt P. et al.
    ///
    /// Trussness is stored as `u8`s, see [`new_tx`](Self::new_tx) for other widths.
    ///
    /// # Arguments
    ///
    /// * `g` --- the  [`GraphMemoryMap`] instance in which k-truss decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_tx(g)
    }

    /// Searches for a previously cached result, and if not found performs k-truss decomposition as described in ["Bounds and algorithms for graph trusses"](https://doi.org/10.48550/arXiv.1806.05523) by Burkhardt P. et al.
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

}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount>
    AlgoBurkhardtEtAl<'a, N, E, Ix, Tx>
{
    /// Performs k-truss decomposition, storing trussness as `Tx` integers.
    pub fn new_tx(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut burkhardt_et_al = Self::new_no_compute(g)?;
        let proc_mem = burkhardt_et_al.init_cache_mem()?;

        burkhardt_et_al.compute_with_proc_mem(proc_mem)?;

        Ok(burkhardt_et_al)
    }

    /// Searches for a previously cached `Tx` result, computing it if not found.
    pub fn get_or_compute_tx(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t_fn = g.build_cache_filename(CacheFile::KTrussBEA, None)?;
        if Path::new(&t_fn).exists() {
            // a cache of another graph (version) or trussness width would have another length
            if let Ok(k_trusses) = AbstractedProceduralMemoryMut::<Tx>::from_file_name(&t_fn)
                && k_trusses.as_slice().len() == g.width()
            {
                return Ok(Self {
                    g,
                    k_trusses,
                    directed: None,
                });
            }
        }
        Self::new_tx(g)
    }

    /// Performs k-truss decomposition as described in ["Bounds and algorithms for graph trusses"](https://doi.org/10.48550/arXiv.1806.05523)
    /// by Burkhardt P. et al. from precomputed `triangles` of `g` (e.g. counted on a GPU, with
    /// [`Triangles::new_gpu`]), which are consumed: their counts are used as working memory.
    ///
    /// [`Triangles::new_gpu`]: ../triangles/struct.Triangles.html#method.new_gpu
    pub fn new_with_triangles(
        g: &'a GraphMemoryMap<N, E, Ix>,
        triangles: Triangles<'a, N, E, Ix, Tx>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let burkhardt_et_al = Self::new_no_compute(g)?;
        let proc_mem = burkhardt_et_al.init_cache_mem()?;
        burkhardt_et_al.compute_with_triangles(proc_mem, triangles)?;
        Ok(burkhardt_et_al)
    }






}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount>
    AlgoBurkhardtEtAl<'a, N, E, Ix, Tx>
{
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, None)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, None)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryBurkhardtEtAl<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryBurkhardtEtAl<Ix>, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBurkhardtEtAl<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBurkhardtEtAl<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.width()
    }


    fn init_cache_mem_impl(&self) -> Result<ProceduralMemoryBurkhardtEtAl<Ix>, Box<dyn std::error::Error>> {
        let mem = self.scratch_mem()?;
        // pre-initialize the memmapped files if they don't exist
        let _edge_reciprocal = self.g.edge_reciprocal()?;
        let _edge_out = self.g.edge_over()?;
        Ok(mem)
    }

    fn compute_with_proc_mem_impl(
        &self,
        proc_mem: ProceduralMemoryBurkhardtEtAl<Ix>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_triangles(proc_mem, Triangles::new_tx(self.g)?)
    }


    /// Decomposes the (undirected) graph from its `triangles`, whose counts are used (and
    /// consumed) as working memory.
    fn compute_with_triangles(
        &self,
        proc_mem: ProceduralMemoryBurkhardtEtAl<Ix>,
        mut triangles: Triangles<'a, N, E, Ix, Tx>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.check_triangles_len(triangles.all_triangles().len())?;
        let edge_reciprocal = self.g.edge_reciprocal()?;
        let index = TriangleIndex::Undirected {
            offsets: SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size()),
            neighbours: SharedSlice::<Ix>::new(self.g.neighbours_ptr(), self.g.width()),
            reciprocal: edge_reciprocal.shared_slice(),
        };
        self.peel::<true>(proc_mem, triangles.triangles_shares_slice(), &index, None)?;
        triangles.drop_cache()?;
        Ok(())
    }


}

/// Shared by the decompositions of undirected graphs' triangles and directed graphs'.
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount, Dir: Directedness>
    AlgoBurkhardtEtAl<'a, N, E, Ix, Tx, Dir>
{
    /// The working memory of a decomposition.
    fn scratch_mem(
        &self,
    ) -> Result<ProceduralMemoryBurkhardtEtAl<Ix>, Box<dyn std::error::Error>> {
        let edge_count = self.g.width();
        let edge_count2 = self.g.width() * 2;
        let file = |seq| {
            self.g
                .build_cache_filename(cache_file(self.directed), Some(seq))
        };

        let edge_list = SharedSliceMut::<Ix>::abst_mem_mut(&file(0)?, edge_count, true)?;
        let edge_index = SharedSliceMut::<Ix>::abst_mem_mut(&file(1)?, edge_count, true)?;
        let stack = SharedSliceMut::<(Ix, Ix)>::abst_mem_mut(&file(2)?, edge_count2, true)?;


        Ok((edge_list, edge_index, stack))
    }

    fn check_triangles_len(&self, len: usize) -> Result<(), Box<dyn std::error::Error>> {
        if len != self.g.width() {
            return Err(format!(
                "error expected the triangles of a graph with {} edges, got {len}",
                self.g.width(),
            )
            .into());
        }
        Ok(())
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
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix, Dir>,
        directed: Option<DirectedTriangleType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(cache_file(directed), None)?;
        let k_trusses = SharedSliceMut::<Tx>::abst_mem_mut(&out_fn, g.width(), true)?;
        Ok(Self {
            g,
            k_trusses,
            directed,
        })
    }
    /// Algorithms 1 and 2, adjusted for multigraphs and directed triangle types: peels edges by
    /// increasing support (their triangle counts, consumed), storing their trussness. Removed
    /// edges are those with support 0 (sentinel), or those marked in `removed` if given.
    /// `SENTINEL`: removed edges are those with support 0 (undirected graphs), rather than those
    /// marked in `removed` (resolved at compile time, keeping the check out of the inner loop).
    fn peel<const SENTINEL: bool>(
        &self,
        proc_mem: ProceduralMemoryBurkhardtEtAl<Ix>,
        tris: SharedSliceMut<Tx::Atomic>,
        index: &TriangleIndex<'_, 'a, N, E, Ix>,
        mut removed: Option<SharedSliceMut<bool>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let edge_count = self.g.width();

        let neighbours_ptr = SharedSlice::<Ix>::new(self.g.neighbours_ptr(), edge_count);

        // Shared atomic & simple arrays for counts and trussness
        let (edges, edge_index, e_stack) = proc_mem;
        let mut trussness = self.k_trusses.shared_slice();

        thread::scope(|scope| {
            // initializations always uses at least two threads per core
            let threads = self.g.thread_num().max(get_physical() * 2);
            let edge_load = edge_count.div_ceil(threads);

            for tid in 0..threads {
                let mut edges = edges.shared_slice();
                let mut edge_index = edge_index.shared_slice();

                let begin = std::cmp::min(tid * edge_load, edge_count);
                let end = std::cmp::min(begin + edge_load, edge_count);

                scope.spawn(move |_| {
                    for idx in begin..end {
                        // minimum possible trussness
                        *trussness.get_mut(idx) = Tx::from_usize(2);
                        *edges.get_mut(idx) = Ix::new(idx);
                        *edge_index.get_mut(idx) = Ix::new(idx);
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        // Algorithm 1 - adjusted for directed scheme, computed by Triangles
        check_support_range::<Tx>(tris.as_slice())?;
        let mut stack = SharedQueueMut::<(Ix, Ix)>::from_shared_slice(e_stack.shared_slice());

        // Algorithm 2 - sentinel value is 0
        // blank (u64, usize) value
        let mut edges = edges.shared_slice();
        let mut edge_index = edge_index.shared_slice();
        let mut edge_count = edge_count;
        // indexed by trussness (`k + 2`)
        let mut test = vec![0usize; 3];

        // with only some roles selected, an edge without triangles counting towards its support
        // may still be in triangles counting towards others': it's peeled (at level 0) like any
        // other edge, rather than just discarded
        let mut closing = Vec::new();
        let mut k = if index.symmetric() { 1 } else { 0 };
        while edge_count > 0 {
            let mut idx = 0;
            while idx < edge_count {
                let edge_offset = edge_index.get(idx).index();
                let t_count = tris.get(edge_offset).count();
                if t_count == k {
                    // at level 0, only edges in some triangle can matter: the others are left to
                    // the next scan, which removes them (as edges without support, trussness 2)
                    if k == 0 && !index.may_be_in_triangle(edge_offset) {
                        idx += 1;
                        continue;
                    }
                    let u = index.source(edge_offset);
                    stack.push((Ix::new(u), Ix::new(edge_offset)));
                    idx = match idx.overflowing_add(1) {
                        (r, false) => r,
                        _ => {
                            return Err(format!("error overflow add to idx ({idx} + 1)",).into());
                        }
                    };
                } else if t_count == 0 {
                    edge_count = match edge_count.overflowing_sub(1) {
                        (r, false) => r,
                        _ => {
                            return Err(format!(
                                "error overflow add to edge_count ({edge_count} + 1)",
                            )
                            .into());
                        }
                    };

                    let e_index = edge_index.get(edge_count).index();
                    let r_index = edges.get(edge_offset).index();
                    *edge_index.get_mut(edge_count) = *edge_index.get(r_index);
                    *edge_index.get_mut(r_index) = Ix::new(e_index);
                    *edges.get_mut(edge_offset) = Ix::new(edge_count);
                    *edges.get_mut(e_index) = Ix::new(r_index);
                    // store edge trussness
                    *trussness.get_mut(edge_offset) = Tx::from_usize(k + 1);
                    if k == 1 {
                        test[2] += 1;
                    }
                    continue;
                } else {
                    idx = match idx.overflowing_add(1) {
                        (r, false) => r,
                        _ => {
                            return Err(format!("error overflow add to idx ({idx} + 1)",).into());
                        }
                    };
                }
            }

            if test.len() <= k + 2 {
                test.resize(k + 3, 0);
            }
            while let Some((u, offset)) = stack.pop() {
                let u = u.index();
                let offset = offset.index();
                tris.get(offset).set_count(0);
                if !SENTINEL && let Some(removed) = removed.as_mut() {
                    *removed.get_mut(offset) = true;
                }
                // by definition any non-trivial subgraph is at least a 2-truss: other
                // values go up from there.
                test[k + 2] += 1;
                let v = neighbours_ptr.get(offset).index();

                index.closing(u, v, &mut closing);
                let alive = |e: usize| {
                    if SENTINEL {
                        tris.get(e).count() != 0
                    } else {
                        removed.as_ref().is_none_or(|removed| !*removed.get(e))
                    }
                };
                for c in closing.iter() {
                    if alive(c.f) && alive(c.g) {
                        // (undirected triangles always count)
                        if (SENTINEL || c.g_counts) && tris.get(c.g).sub_count(1) == k + 1 {
                            stack.push((Ix::new(index.source(c.g)), Ix::new(c.g)));
                        }
                        if (SENTINEL || c.f_counts) && tris.get(c.f).sub_count(1) == k + 1 {
                            stack.push((Ix::new(index.source(c.f)), Ix::new(c.f)));
                        }
                    }
                }
            }
            k += 1;
        }
        let env_verbose_val = std::env::var("BRUIJNX_VERBOSE").unwrap_or_else(|_| "0".to_string());
        let verbose: bool = env_verbose_val == "1";
        if verbose {
            let mut max = 0;
            test.iter().enumerate().for_each(|(i, v)| {
                if *v != 0 && i > max {
                    max = i;
                }
            });
            test.resize(max + 1, 0);
            println!("k-trussness {:?}", test);
        }
        self.k_trusses.flush_async()?;

        // cleanup cache (edge list, edge index, stack and removed flags)
        self.g.remove_tmp_files(cache_file(self.directed), 0..4)?;

        Ok(())
    }
}

/// Decompositions of a directed graph's triangles of a given type.
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> AlgoBurkhardtEtAl<'a, N, E, Ix, u8, Directed> {
    /// Performs k-truss decomposition of a directed graph, over its triangles of type
    /// `triangle_type` (an edge is in the k-truss if it's in at least `k - 2` triangles, in a
    /// selected role, of the k-truss). Trussness is stored as `u8`s, see
    /// [`new_directed_tx`](Self::new_directed_tx) for other widths.
    pub fn new_directed(
        g: &'a DiGraph<N, E, Ix>,
        triangle_type: DirectedTriangleType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_directed_tx(g, triangle_type)
    }
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, Tx: TriangleCount>
    AlgoBurkhardtEtAl<'a, N, E, Ix, Tx, Directed>
{
    /// Performs k-truss decomposition of a directed graph, over its triangles of type
    /// `triangle_type`, storing trussness as `Tx` integers.
    pub fn new_directed_tx(
        g: &'a DiGraph<N, E, Ix>,
        triangle_type: DirectedTriangleType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // the merged adjacency serves both counting and peeling
        let adjacency = DirectedAdjacency::build(g, g.thread_num())?;
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
            let burkhardt_et_al = Self::new_no_compute_impl(g, Some(triangle_type))?;
            let proc_mem = burkhardt_et_al.scratch_mem()?;
            burkhardt_et_al.compute_directed_with_triangles(
                proc_mem,
                triangles,
                Some((&adjacency, in_triangle.as_ref().map(|f| f.shared_slice()))),
            )?;
            Ok(burkhardt_et_al)
        })();
        adjacency.cleanup()?;
        result
    }
    /// Performs k-truss decomposition of a directed graph from its precomputed `triangles` (of
    /// their type), which are consumed: their counts are used as working memory.
    pub fn new_directed_with_triangles(
        g: &'a DiGraph<N, E, Ix>,
        triangles: DirectedTriangles<'a, N, E, Ix, Tx>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let burkhardt_et_al = Self::new_no_compute_impl(g, Some(triangles.triangle_type()))?;
        let proc_mem = burkhardt_et_al.scratch_mem()?;
        burkhardt_et_al.compute_directed_with_triangles(proc_mem, triangles, None)?;
        Ok(burkhardt_et_al)
    }
    /// Decomposes the directed graph from its `triangles`, whose counts are used (and consumed)
    /// as working memory.
    fn compute_directed_with_triangles(
        &self,
        proc_mem: ProceduralMemoryBurkhardtEtAl<Ix>,
        mut triangles: DirectedTriangles<'a, N, E, Ix, Tx>,
        adjacency: Option<(
            &DirectedAdjacency<'a, N, E, Ix>,
            Option<SharedSliceMut<portable_atomic::AtomicBool>>,
        )>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.check_triangles_len(triangles.all_triangles().len())?;
        // the caller's merged adjacency, or one built (and removed) here
        let built = match adjacency {
            Some(_) => None,
            None => Some(DirectedAdjacency::build(self.g, self.g.thread_num())?),
        };
        let (adjacency, in_triangle) = match (adjacency, built.as_ref()) {
            (Some((adjacency, in_triangle)), _) => (adjacency, in_triangle),
            (None, Some(built)) => (built, None),
            (None, None) => return Err("error no merged adjacency".into()),
        };
        let index = TriangleIndex::Directed {
            adjacency,
            in_triangle,
            offsets: SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size()),
            roles: triangles.triangle_type().roles(),
        };
        // with only some roles selected, an edge may be in triangles none of which count towards
        // its support, so removed edges are tracked apart (support 0 doesn't mean removed)
        let r_fn = self
            .g
            .build_cache_filename(cache_file(self.directed), Some(3))?;
        let removed = SharedSliceMut::<bool>::abst_mem_mut(&r_fn, self.g.width(), true)?;
        let result = self.peel::<false>(
            proc_mem,
            triangles.triangles_shares_slice(),
            &index,
            Some(removed.shared_slice()),
        );
        if let Some(built) = built {
            built.cleanup()?;
        }
        result?;
        triangles.drop_cache()?;
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
                    fn [<k_trusses_burkhardt_et_al_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
                        generic_test($path)
                    }
                }
            )*
        }
    }

    fn generic_test<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn std::error::Error>> {
        // a private copy: tests on the same fixture may run concurrently
        let (graph, _cache) = crate::graph::test_utils::open_fixture(path.as_ref(), Some(16))?;
        let burkhardt_et_al_k_trusses = AlgoBurkhardtEtAl::new(&graph)?;

        verify_k_trusses(&graph, burkhardt_et_al_k_trusses.k_trusses())
    }

    // generate test cases from dataset
    graph_tests! {
        ggcat_1_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_1_5.lz4"),
        ggcat_2_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_2_5.lz4"),
        ggcat_3_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_3_5.lz4"),
        ggcat_4_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_4_5.lz4"),
        // ggcat_5_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_5_5.lz4"),
        // ggcat_6_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_6_5.lz4"),
        // ggcat_7_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_7_5.lz4"),
        // ggcat_8_5 => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggcat/random_graph_8_5.lz4"),
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
