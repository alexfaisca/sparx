pub mod approx_dirichlet_hkpr;
pub mod directed_louvain;
pub mod gve_louvain;
pub mod hk_relax;
pub mod label_propagation;
pub mod spectral;

use crate::graph;
use portable_atomic::{AtomicUsize, Ordering};

/// Whether Louvain ([`DirectedLouvain`](directed_louvain::DirectedLouvain), and
/// [`AlgoGVELouvain`](gve_louvain::AlgoGVELouvain)) refines its communities before aggregating
/// them. Either way, communities are split into their connected parts at the end, so none is
/// disconnected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Refinement {
    /// No: Louvain's, aggregating communities as local moving finds them (some may be
    /// internally disconnected, a node holding one together having moved out of it, until
    /// they're split at the end). The default.
    #[default]
    None,
    /// Leiden's, as GVE-Leiden's (Sahu's ["GVE-Leiden: Fast Leiden Algorithm for Community
    /// Detection in Shared Memory Setting"](https://doi.org/10.48550/arXiv.2312.13936)): each
    /// community local moving finds is split into the sub-communities its nodes form when,
    /// starting alone, each joins the neighbouring sub-community of its community it gains most
    /// modularity joining, if it's still alone (once, in parallel); sub-communities, connected,
    /// are aggregated, starting the next pass in their communities. Communities are then
    /// (weakly) connected, at the cost of a sweep more by pass.
    Leiden,
}

/// Splits communities into their (weakly) connected parts: `comm` (any labels smaller than
/// `comm.len()`) becomes the parts, numbered `0..count` by their smallest node, and their number
/// is returned. Parts are found by a lock-free union-find over the arcs within communities
/// (`offsets`/`neighbours`, a CSR), in parallel.
///
/// Splitting a disconnected community never lowers modularity, directed or not: no arc is
/// between its parts, and their expected arcs (`Σ_out Σ_in`, or `Σ²`) add up to less than the
/// whole's. Louvain's communities, then, are only made better by it, and connected.
pub(crate) fn split_disconnected<Ix: graph::IndexType, L: graph::IndexType>(
    threads: usize,
    offsets: &[Ix],
    neighbours: &[Ix],
    comm: &mut [L],
) -> Result<usize, Box<dyn std::error::Error>> {
    let n = comm.len();
    let threads = threads.max(1);
    let parent: Vec<AtomicUsize> = (0..n).map(AtomicUsize::new).collect();
    let find = |mut x: usize| loop {
        let p = parent[x].load(Ordering::Relaxed);
        if p == x {
            return x;
        }
        let grandparent = parent[p].load(Ordering::Relaxed);
        if grandparent != p {
            let _ =
                parent[x].compare_exchange(p, grandparent, Ordering::Relaxed, Ordering::Relaxed);
        }
        x = p;
    };
    let in_parallel = |f: &(dyn Fn(usize, usize) + Sync)| {
        let load = n.div_ceil(threads);
        crossbeam::thread::scope(|scope| {
            for tid in 0..threads {
                let begin = (tid * load).min(n);
                let end = (begin + load).min(n);
                scope.spawn(move |_| f(begin, end));
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })
    };
    {
        let comm = &*comm;
        in_parallel(&|begin, end| {
            for u in begin..end {
                let cu = comm[u].index();
                for v in &neighbours[offsets[u].index()..offsets[u + 1].index()] {
                    let v = v.index();
                    if v == u || comm[v].index() != cu {
                        continue;
                    }
                    // link the larger root to the smaller: each part's root is its smallest node
                    loop {
                        let (ru, rv) = (find(u), find(v));
                        if ru == rv {
                            break;
                        }
                        let (high, low) = (ru.max(rv), ru.min(rv));
                        if parent[high]
                            .compare_exchange(high, low, Ordering::AcqRel, Ordering::Relaxed)
                            .is_ok()
                        {
                            break;
                        }
                    }
                }
            }
        })?;
    }
    // parts numbered by their smallest node (their root): the roots before it, by range
    let load = n.div_ceil(threads);
    let counts: Vec<usize> = crossbeam::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let (begin, end) = ((tid * load).min(n), ((tid + 1) * load).min(n));
                let find = &find;
                scope.spawn(move |_| (begin..end).filter(|&u| find(u) == u).count())
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
    .map_err(|e| -> Box<dyn std::error::Error> { format!("{e:?}").into() })?;
    let mut starts = Vec::with_capacity(threads);
    let mut count = 0;
    for c in counts {
        starts.push(count);
        count += c;
    }
    // each node's root, each root's number, then each node's part's
    let mut roots = vec![0usize; n];
    {
        let r = crate::shared_slice::SharedSliceMut::<usize>::new(roots.as_mut_ptr(), n);
        in_parallel(&|begin, end| {
            let mut r = r;
            for u in begin..end {
                *r.get_mut(u) = find(u);
            }
        })?;
    }
    let numbers: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
    {
        let roots = &roots;
        let numbers = &numbers;
        let starts = &starts;
        in_parallel(&|begin, end| {
            let mut id = if begin < end { starts[begin / load] } else { 0 };
            for u in begin..end {
                if roots[u] == u {
                    numbers[u].store(id, Ordering::Relaxed);
                    id += 1;
                }
            }
        })?;
    }
    let out = crate::shared_slice::SharedSliceMut::<L>::new(comm.as_mut_ptr(), n);
    in_parallel(&|begin, end| {
        let mut out = out;
        for u in begin..end {
            *out.get_mut(u) = L::new(numbers[roots[u]].load(Ordering::Relaxed));
        }
    })?;
    Ok(count)
}
