pub mod burkhardt_et_al;
pub mod clustering_coefficient;
pub mod count;
pub mod directed;
pub mod pkt;
pub mod triangles;

use crate::graph::{self, IndexType};
use crate::shared_slice::{SharedSlice, SharedSliceMut};

use count::{TriangleCount, TriangleCounter};
use directed::{DirectedAdjacency, IN, MIDDLEMAN, OUT, role};

#[cfg(test)]
mod matrix_test;
#[cfg(test)]
mod oracle_test;

/// Checks every edge's trussness (its triangle count, plus 2, at most) can be represented as a
/// `Tx`, the k-truss decompositions' output.
pub(crate) fn check_support_range<Tx: TriangleCount>(
    triangles: &[Tx::Atomic],
) -> Result<(), Box<dyn std::error::Error>> {
    let max_support = Tx::MAX - 2;
    match triangles.iter().map(|t| t.count()).max() {
        Some(max) if max > max_support => Err(format!(
            "error an edge is in {max} triangles, but trussness above {} (more than \
             {max_support} triangles) can't be represented ({})",
            Tx::MAX,
            std::any::type_name::<Tx>()
        )
        .into()),
        _ => Ok(()),
    }
}

/// A triangle of an edge `e`: its two other edges `f` and `g`, and whether the triangle counts
/// towards their support (always, for undirected graphs; when their role in it is selected, for
/// directed ones).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Closing {
    pub(crate) f: usize,
    pub(crate) g: usize,
    pub(crate) f_counts: bool,
    pub(crate) g_counts: bool,
}

/// What k-truss decompositions need of a graph's triangles: each edge's source, and the
/// triangles of an edge.
pub(crate) enum TriangleIndex<'b, 'a, N: graph::N, E: graph::E, Ix: IndexType> {
    /// Undirected (symmetric) graph: every edge has a reciprocal.
    Undirected {
        offsets: SharedSlice<Ix>,
        neighbours: SharedSlice<Ix>,
        reciprocal: SharedSlice<Ix>,
    },
    /// Directed graph, counting triangles in the `roles` selected (and, if known, which edges are
    /// in any triangle at all).
    Directed {
        adjacency: &'b DirectedAdjacency<'a, N, E, Ix>,
        offsets: SharedSlice<Ix>,
        roles: u8,
        in_triangle: Option<SharedSliceMut<portable_atomic::AtomicBool>>,
    },
}

impl<N: graph::N, E: graph::E, Ix: IndexType> TriangleIndex<'_, '_, N, E, Ix> {
    /// Whether every triangle counts towards the support of all its edges or of none (always for
    /// undirected graphs; for directed ones, unless only some feed-forward roles are selected).
    /// If so, an edge without support is in no triangle that counts towards others' either.
    #[inline(always)]
    pub(crate) fn symmetric(&self) -> bool {
        match self {
            TriangleIndex::Undirected { .. } => true,
            TriangleIndex::Directed { roles, .. } => {
                let feed_forward = roles & (OUT | IN | MIDDLEMAN);
                feed_forward == 0 || feed_forward == OUT | IN | MIDDLEMAN
            }
        }
    }

    /// Whether edge `e` may be in a triangle (only known not to be, for directed graphs whose
    /// edges in triangles were flagged): edges in none can't change others' support.
    #[inline(always)]
    pub(crate) fn may_be_in_triangle(&self, e: usize) -> bool {
        match self {
            TriangleIndex::Directed {
                in_triangle: Some(in_triangle),
                ..
            } => in_triangle.get(e).load(portable_atomic::Ordering::Relaxed),
            _ => true,
        }
    }

    /// The node whose neighbour list holds edge `e`.
    #[inline(always)]
    pub(crate) fn source(&self, e: usize) -> usize {
        match self {
            TriangleIndex::Undirected {
                neighbours,
                reciprocal,
                ..
            } => neighbours.get(reciprocal.get(e).index()).index(),
            TriangleIndex::Directed { offsets, .. } => {
                offsets.as_slice().partition_point(|o| o.index() <= e) - 1
            }
        }
    }

    /// Fills `triangles` with the triangles of edge `u -> v` (one per combination of parallel
    /// edges), but those counting towards neither of their other edges' support.
    pub(crate) fn closing(&self, u: usize, v: usize, triangles: &mut Vec<Closing>) {
        triangles.clear();
        // a self-loop is in no triangle (merging its node's neighbours with themselves would
        // pair every neighbour with itself)
        if u == v {
            return;
        }
        match self {
            TriangleIndex::Undirected {
                offsets,
                neighbours,
                reciprocal,
            } => {
                // merge `u`'s and `v`'s sorted neighbours: for every common neighbour `w`, the
                // closing edges are `v -> w` and `w -> u`, once per combination of parallel edges
                let (mut p, p_end) = (offsets.get(u).index(), offsets.get(u + 1).index());
                let (mut q, q_end) = (offsets.get(v).index(), offsets.get(v + 1).index());
                while p < p_end && q < q_end {
                    let a = neighbours.get(p).index();
                    let b = neighbours.get(q).index();
                    match a.cmp(&b) {
                        std::cmp::Ordering::Less => p += 1,
                        std::cmp::Ordering::Greater => q += 1,
                        std::cmp::Ordering::Equal => {
                            // runs of parallel edges to the common neighbour
                            let mut p_run = p;
                            while p_run < p_end && neighbours.get(p_run).index() == a {
                                p_run += 1;
                            }
                            let mut q_run = q;
                            while q_run < q_end && neighbours.get(q_run).index() == a {
                                q_run += 1;
                            }
                            if a != u && a != v {
                                for u_w in p..p_run {
                                    for v_w in q..q_run {
                                        triangles.push(Closing {
                                            f: v_w,
                                            g: reciprocal.get(u_w).index(),
                                            f_counts: true,
                                            g_counts: true,
                                        });
                                    }
                                }
                            }
                            p = p_run;
                            q = q_run;
                        }
                    }
                }
            }
            TriangleIndex::Directed {
                adjacency, roles, ..
            } => {
                let (mut p, p_end) = (adjacency.range(u).start, adjacency.range(u).end);
                let (mut q, q_end) = (adjacency.range(v).start, adjacency.range(v).end);
                while p < p_end && q < q_end {
                    let (uw, vw) = (adjacency.entry(p), adjacency.entry(q));
                    match uw.node.cmp(&vw.node) {
                        std::cmp::Ordering::Less => p += 1,
                        std::cmp::Ordering::Greater => q += 1,
                        std::cmp::Ordering::Equal => {
                            let w = uw.node.index();
                            if w != u && w != v {
                                // `f` between `u` and `w`, `g` between `v` and `w`, either way
                                for (f_run, f_u_to_w) in
                                    [(adjacency.out_run(u, uw), true), (adjacency.in_run(u, uw), false)]
                                {
                                    for (g_run, g_v_to_w) in [
                                        (adjacency.out_run(v, vw), true),
                                        (adjacency.in_run(v, vw), false),
                                    ] {
                                        // the triangle's third edge is `u -> v`
                                        let f_role = if f_u_to_w {
                                            role(true, !g_v_to_w)
                                        } else {
                                            role(!g_v_to_w, true)
                                        };
                                        let g_role = if g_v_to_w {
                                            role(false, !f_u_to_w)
                                        } else {
                                            role(!f_u_to_w, false)
                                        };
                                        let (f_counts, g_counts) =
                                            (roles & f_role != 0, roles & g_role != 0);
                                        if !f_counts && !g_counts {
                                            continue;
                                        }
                                        for f in f_run.edges() {
                                            for g in g_run.edges() {
                                                triangles.push(Closing {
                                                    f,
                                                    g,
                                                    f_counts,
                                                    g_counts,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                            p += 1;
                            q += 1;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
use _verify::verify_k_trusses;

#[cfg(test)]
mod _verify {
    use super::count::TriangleCount;
    use crate::graph::{self, GraphMemoryMap, IndexType};
    use crate::shared_slice::SharedSlice;

    use crossbeam::thread;
    use num_cpus::get_physical;

    /// Verifies `edge_trussness` (as given by the decompositions: the largest `k` such that the
    /// edge is in the `k`-truss) is a valid k-truss decomposition of the (undirected) `graph`:
    /// symmetric, and for every edge of trussness `k`, `k - 2` of its triangles (made of edges of
    /// trussness at least `k`) exist (membership), but not `k - 1` made of edges of trussness at
    /// least `k + 1` (maximality).
    #[allow(dead_code)]
    pub(super) fn verify_k_trusses<N: graph::N, E: graph::E, Ix: IndexType, Tx: TriangleCount>(
        graph: &GraphMemoryMap<N, E, Ix>,
        edge_trussness: &[Tx],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threads = get_physical() * 2;
        let node_count = graph.size();
        let node_load = node_count.div_ceil(threads);
        // trussness length == graph width
        if edge_trussness.len() != graph.width() {
            return Err(format!(
                "k-trusses invalid: graph width is {} but trussness has length {}",
                graph.width(),
                edge_trussness.len()
            )
            .into());
        }
        if graph.width() == 0 {
            return Ok(());
        }

        let index_ptr = SharedSlice::<Ix>::new(graph.offsets_ptr(), graph.offsets_size());
        let neighbours_ptr = SharedSlice::<Ix>::new(graph.neighbours_ptr(), graph.width());
        let offset = |u: usize| index_ptr.get(u).index();
        let edge_reciprocal = graph.edge_reciprocal()?;
        let edge_out = graph.edge_over()?;
        let trussness = |e: usize| edge_trussness[e].to_usize();

        // triangles `(u, v, w)` whose other two edges have trussness at least `tau`
        let count_support = |u: usize,
                             u_offset: usize,
                             v: usize,
                             v_offset: usize,
                             tau: usize|
         -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
            let mut n_u = graph.neighbours(u).map_err(|e| e.to_string())?.enumerate();
            let mut n_v = graph.neighbours(v).map_err(|e| e.to_string())?.enumerate();
            let mut u_w = n_u.next();
            let mut v_w = n_v.next();
            let mut cnt = 0;

            while let (Some((u_w_idx, u_w_node)), Some((v_w_idx, v_w_node))) = (u_w, v_w) {
                match u_w_node.cmp(&v_w_node) {
                    std::cmp::Ordering::Less => u_w = n_u.next(),
                    std::cmp::Ordering::Greater => v_w = n_v.next(),
                    std::cmp::Ordering::Equal => {
                        let w = u_w_node;
                        if w != u
                            && w != v
                            && trussness(u_offset + u_w_idx) >= tau
                            && trussness(v_offset + v_w_idx) >= tau
                        {
                            cnt += 1;
                        }
                        u_w = n_u.next();
                        v_w = n_v.next();
                    }
                }
            }
            Ok(cnt)
        };

        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut res = vec![];
            for tid in 0..threads {
                let eo = edge_out.shared_slice();
                let er = edge_reciprocal.shared_slice();

                let start = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(start + node_load, node_count);

                res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        for u in start..end {
                            let u_offset = offset(u);
                            for edge_idx in eo.get(u).index()..offset(u + 1) {
                                let reciprocal_idx = er.get(edge_idx).index();

                                let k_uv = trussness(edge_idx);
                                let k_vu = trussness(reciprocal_idx);

                                // test for trussness symmetry
                                if k_uv != k_vu {
                                    return Err(format!(
                                        "k-trusses invalid: for node {u}: t(u->v)={k_uv}, t(v->u)={k_vu}"
                                    )
                                    .into());
                                }
                                let k = k_uv;

                                let v = neighbours_ptr.get(edge_idx).index();
                                let v_offset = offset(v);
                                // membership: support within edges of trussness >= k is >= k - 2
                                let support_k = count_support(u, u_offset, v, v_offset, k)?;
                                if support_k + 2 < k {
                                    return Err(format!(
                                        "Edge ({u},{v}) fails {k}-truss membership: support_>={k} = {support_k} < {}",
                                        k - 2
                                    )
                                    .into());
                                }
                                // maximality: support within edges of trussness >= k + 1 is < k - 1
                                let support_k1 = count_support(u, u_offset, v, v_offset, k + 1)?;
                                if support_k1 + 1 >= k {
                                    return Err(format!(
                                        "Edge ({u},{v}) fails {k}-truss maximality: support_>={} = {support_k1} (should be < {})",
                                        k + 1,
                                        k - 1
                                    )
                                    .into());
                                }
                            }
                        }
                        Ok(())
                    },
                ));
            }
            // join results
            for r in res {
                let joined_r = match r.join() {
                    Ok(b) => b,
                    Err(e) => {
                        return Err(format!("{:?}", e).into());
                    }
                };
                joined_r.map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
    }
}
