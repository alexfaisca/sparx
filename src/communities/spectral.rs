//! Spectral clustering of large graphs, by polynomial filtering: [`SpectralClustering`].
//!
//! Spectral clustering embeds each node by the `k` eigenvectors of the normalized Laplacian
//! of least eigenvalues, and clusters the embedding with k-means. Finding the eigenvectors is
//! what doesn't scale. This finds an equivalent embedding without them, as *compressive
//! spectral clustering* does (Tremblay N., Puy G., Gribonval R. and Vandergheynst P.,
//! ["Compressive spectral clustering"](https://arxiv.org/abs/1602.02018)): a low-pass graph
//! filter, keeping the `k` lowest frequencies, applied to `d = O(log k)` random signals gives
//! every node `d` features whose distances approximate the spectral embedding's. Every step is a
//! product of the graph's (sparse) matrix with a block of vectors, in parallel:
//!
//! 1. the cut-off frequency `λ_k` (the `k`-th least eigenvalue) is estimated by the *kernel
//!    polynomial method* (Silver R. N. and Röder H.; Lin L., Saad Y. and Yang C.,
//!    ["Approximating spectral densities of large matrices"](https://doi.org/10.1137/130934283)):
//!    the Chebyshev moments of random `±1` signals give the number of eigenvalues below any
//!    `λ`, at once;
//! 2. the ideal low-pass filter `1[L <= λ_k]` is approximated by a Chebyshev polynomial (Jackson
//!    damped, so that it doesn't ring) and applied to `d` Gaussian signals: rows, normalized,
//!    are the nodes' features;
//! 3. k-means (k-means++, then Lloyd's) clusters a uniform sample of the nodes, and every node
//!    joins the cluster of the nearest centroid, then a few Lloyd's iterations over all nodes
//!    refine them (rather than the paper's interpolation, as every node's features are at
//!    hand anyway).
//!
//! The Laplacian is the *regularized* one (Amini A. A. et al., ["Pseudo-likelihood methods for
//! community detection in large sparse networks"](https://doi.org/10.1214/13-AOS1138); Qin T. and
//! Rohe K., ["Regularized spectral clustering under the degree-corrected stochastic
//! blockmodel"](https://arxiv.org/abs/1309.4111)), `L = I - D_τ^-1/2 A D_τ^-1/2` with `D_τ = D + τI`
//! (`τ` the mean degree by default): on sparse graphs, the least eigenvectors of the plain one
//! pick out tiny components and dangling trees rather than clusters.
//!
//! Results are the same whatever the threads: random signals are hashes of the seed, a node and
//! a column; every row of a product is computed by one thread, in arc order; sums over nodes are
//! of fixed blocks, added in order.

use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicUsize, Ordering};

type Error = Box<dyn std::error::Error>;

/// The matrix whose low eigenvectors [`SpectralClustering`] embeds nodes by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpectralMatrix {
    /// The regularized normalized Laplacian `I - D_τ^-1/2 A D_τ^-1/2` (see the [module
    /// docs](self)), its `k` least eigenvalues'. Any graph, weighted or not; `k` must be given.
    #[default]
    RegularizedLaplacian,
    /// The Bethe Hessian `H(r) = (r^2 - 1) I - r A + D` (Saade A., Krzakala F. and Zdeborová L.,
    /// ["Spectral clustering of graphs with the Bethe Hessian"](https://arxiv.org/abs/1406.1880)),
    /// `r` the square root of the mean excess degree (`Σ d^2 / Σ d - 1`), its negative
    /// eigenvalues': on sparse graphs, it finds clusters down to the theoretical detectability
    /// threshold, where the Laplacian's eigenvectors are already lost in its bulk (on a planted
    /// partition of a million nodes of mean degree 16, 64 blocks, 60% of arcs across them, it
    /// found them with an NMI of 0.83, where the Laplacian's found nothing); their number is
    /// roughly that of its negative eigenvalues (see [`SpectralOptions::k`]).
    /// Unweighted graphs only. Its spectrum's width is about the largest degree's `(1 + r)`
    /// times: on graphs of very unequal degrees, it takes more moments and a higher degree
    /// filter to resolve.
    BetheHessian,
}

/// How [`SpectralClustering`] clusters. [`SpectralOptions::new`]`(k)` gives the defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralOptions {
    /// Clusters to find (at most: k-means may leave some empty); `None` to estimate them, with
    /// the [Bethe Hessian](SpectralMatrix::BetheHessian), as its clearly negative eigenvalues'
    /// count (see [`SpectralClustering::k_estimate`]): a rough estimate, which on large graphs
    /// tends to count a few eigenvalues of the bulk too, and whose noise grows as `sqrt(k)`.
    pub k: Option<usize>,
    /// The matrix nodes are embedded by.
    pub matrix: SpectralMatrix,
    /// The most clusters an estimated `k` may be (an error, beyond: k-means of so many clusters
    /// is likely a mistake).
    pub max_k: usize,
    /// `τ`, added to every degree (see the [module docs](self)); `None` for the mean degree.
    /// The regularized Laplacian's only.
    pub regularization: Option<f64>,
    /// The Chebyshev polynomial's degree for the low-pass filter: products of the matrix with
    /// the block of features. Higher degrees cut sharper at `λ_k`, which takes more on graphs
    /// whose low eigenvalues crowd together (long paths).
    pub filter_degree: usize,
    /// The Chebyshev moments estimating `λ_k` (half as many products): their number sets how
    /// finely eigenvalues are told apart.
    pub moments: usize,
    /// The random `±1` signals the eigenvalue counts are averaged over.
    pub probes: usize,
    /// Features per node (`d`); `None` for `⌈4 ln n'⌉`, `n'` the sample's size.
    pub features: Option<usize>,
    /// Nodes k-means++ and Lloyd's iterations first cluster (`n'`); `None` for
    /// `max(20k, 2k ln k)`, or every node, if fewer.
    pub sample: Option<usize>,
    /// k-means runs on the sample, from different seeds, the least inertia winning.
    pub restarts: usize,
    /// Lloyd's iterations over all nodes, after they join the nearest centroids.
    pub refine: usize,
    /// The seed of every random choice.
    pub seed: u64,
}

impl SpectralOptions {
    /// The defaults, for `k` clusters, by the regularized Laplacian.
    pub fn new(k: usize) -> Self {
        Self {
            k: Some(k),
            matrix: SpectralMatrix::RegularizedLaplacian,
            max_k: 1 << 12,
            regularization: None,
            filter_degree: 64,
            moments: 128,
            probes: 16,
            features: None,
            sample: None,
            restarts: 4,
            refine: 3,
            seed: 0x5eed,
        }
    }

    /// The defaults with the [Bethe Hessian](SpectralMatrix::BetheHessian), for `k` clusters.
    /// Near the detectability threshold, where it pays, its eigenvalues that matter are close to
    /// its bulk: raise [`filter_degree`](Self::filter_degree) and [`moments`](Self::moments) (6
    /// times the defaults found clusters the Laplacian can't).
    pub fn bethe_hessian(k: usize) -> Self {
        Self {
            matrix: SpectralMatrix::BetheHessian,
            ..Self::new(k)
        }
    }
}

/// Spectral clustering of an undirected [`GraphMemoryMap`] (see the [module docs](self)), its
/// clusters numbered by their least node. For a directed graph, cluster its
/// [`to_undirected`](GraphMemoryMap::to_undirected) one. Weighted graphs' weights are the
/// adjacency's entries (finite and non-negative).
///
/// Memory: three blocks of `d` features per node, of `W`s (`f64`, or `f32` for half as much),
/// mapped in the graph's cache while clustering, and the clusters.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
pub struct SpectralClustering<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight = f64> {
    g: &'a Graph<N, E, Ix>,
    communities: AbstractedProceduralMemoryMut<Ix>,
    count: usize,
    lambda_k: f64,
    /// The estimated number of clusters, unrounded, and its standard error, if estimated.
    estimate: Option<(f64, f64)>,
    modularity: f64,
    _w: std::marker::PhantomData<W>,
}

/// Rows a thread takes at a time.
const ROWS: usize = 1024;

/// Runs `f(begin, end)` on `n` rows, in chunks of [`ROWS`] taken by `threads` threads as they
/// finish: `f`'s results, in chunk order (so that their sum is the same whatever the threads).
fn chunks<T: Send>(n: usize, threads: usize, f: &(dyn Fn(usize, usize) -> T + Sync)) -> Vec<T> {
    let count = n.div_ceil(ROWS);
    let threads = threads.clamp(1, count.max(1));
    let next = AtomicUsize::new(0);
    let mut parts: Vec<(usize, T)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                s.spawn(move || {
                    let mut mine = vec![];
                    loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        if c >= count {
                            break mine;
                        }
                        mine.push((c, f(c * ROWS, ((c + 1) * ROWS).min(n))));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    parts.sort_unstable_by_key(|p| p.0);
    parts.into_iter().map(|p| p.1).collect()
}

/// SplitMix64: random bits from a counter.
#[inline(always)]
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Random bits of `seed`, a row and a column.
#[inline(always)]
fn bits(seed: u64, row: usize, col: usize) -> u64 {
    mix(mix(seed ^ mix(row as u64)) ^ col as u64)
}

/// A uniform `(0, 1)` of random bits.
#[inline(always)]
fn uniform(b: u64) -> f64 {
    ((b >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// A standard normal, of `seed`, a row and a column (Box-Muller).
#[inline(always)]
fn gaussian(seed: u64, row: usize, col: usize) -> f64 {
    let (u, v) = (
        uniform(bits(seed, row, 2 * col)),
        uniform(bits(seed, row, 2 * col + 1)),
    );
    (-2. * u.ln()).sqrt() * (2. * std::f64::consts::PI * v).cos()
}

/// `n × b` matrices, row-major, of `W`s, mapped in the cache.
struct Block<W: Weight> {
    mem: AbstractedProceduralMemoryMut<W>,
    b: usize,
}

impl<W: Weight> Block<W> {
    fn new(file: &str, n: usize, b: usize) -> Result<Self, Error> {
        Ok(Self {
            mem: SharedSliceMut::<W>::abst_mem_mut(file, (n * b).max(1), true)?,
            b,
        })
    }
    fn shared(&self) -> SharedSliceMut<W> {
        self.mem.shared_slice()
    }
}

/// A [`Block`]'s memory, for chunks of its rows to be written by different threads.
#[derive(Clone, Copy)]
struct Rows<W> {
    ptr: *mut W,
    b: usize,
}

unsafe impl<W: Send> Send for Rows<W> {}
unsafe impl<W: Sync> Sync for Rows<W> {}

impl<W: Weight> Rows<W> {
    fn of(block: &Block<W>) -> Self {
        let mut shared = block.shared();
        Self {
            ptr: shared.as_mut_ptr(),
            b: block.b,
        }
    }

    /// Rows `begin..end`.
    ///
    /// # Safety
    ///
    /// No other reference to these rows may be alive while the slice is, nor may it outlive
    /// the block.
    unsafe fn rows<'r>(self, begin: usize, end: usize) -> &'r mut [W] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.add(begin * self.b), (end - begin) * self.b)
        }
    }
}

/// A graph's matrix `H`, as `M = (H - center) / half`, whose spectrum is in `[-1, 1]` (Chebyshev
/// polynomials' domain): `H_uv = κ σ_u σ_v w_uv` off the diagonal, `H_uu = shift + slope d_u`
/// (`d_u` the arcs' count) on it.
///
/// * The regularized Laplacian, `I - D_τ^-1/2 A D_τ^-1/2`: `σ_u = 1 / sqrt(d_u + τ)` (0 for
///   isolated nodes, without regularization), `κ = -1`, `shift = 1`, `slope = 0`.
/// * The Bethe Hessian: `σ_u = 1`, `κ = -r`, `shift = r^2 - 1`, `slope = 1`.
///
/// `[center - half, center + half]` holds the spectrum: Gershgorin's bounds first, then
/// [`tighten`](Self::tighten)ed by Lanczos' estimates, as the moments resolve eigenvalues in
/// proportion to its width.
struct Operator<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    g: &'a Graph<N, E, Ix>,
    /// `σ`, if not all ones.
    scale: Option<AbstractedProceduralMemoryMut<f64>>,
    kappa: f64,
    shift: f64,
    slope: f64,
    center: f64,
    half: f64,
    threads: usize,
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> Operator<'a, N, E, Ix> {
    fn laplacian(g: &'a Graph<N, E, Ix>, tau: Option<f64>, file: &str) -> Result<Self, Error> {
        let (n, threads) = (g.size(), g.thread_num().max(1));
        // checks the weights finite and non-negative; the mean degree
        let total = g.total_weight()?;
        let tau = tau.unwrap_or(if n > 0 { total / n as f64 } else { 0. });
        if !(tau.is_finite() && tau >= 0.) {
            return Err(format!(
                "error the regularization must be finite and non-negative, not {tau}"
            )
            .into());
        }
        let mut op = Self {
            g,
            scale: None,
            kappa: -1.,
            shift: 1.,
            slope: 0.,
            center: 1.,
            half: 1.,
            threads,
        };
        let scale = SharedSliceMut::<f64>::abst_mem_mut(file, n.max(1), true)?;
        let s = scale.shared_slice();
        chunks(n, threads, &|begin, end| {
            let mut s = s;
            for u in begin..end {
                let d = op.degree(u) + tau;
                *s.get_mut(u) = if d > 0. { 1. / d.sqrt() } else { 0. };
            }
        });
        op.scale = Some(scale);
        Ok(op)
    }

    fn bethe_hessian(g: &'a Graph<N, E, Ix>) -> Result<Self, Error> {
        if E::is_weighted() {
            return Err("error the Bethe Hessian is of unweighted graphs only".into());
        }
        let (n, threads) = (g.size(), g.thread_num().max(1));
        let mut op = Self {
            g,
            scale: None,
            kappa: 0.,
            shift: 0.,
            slope: 1.,
            center: 0.,
            half: 1.,
            threads,
        };
        // Σ d, Σ d^2, and the least and largest degrees
        let parts = chunks(n, threads, &|begin, end| {
            let (mut s1, mut s2, mut lo, mut hi) = (0f64, 0f64, f64::INFINITY, 0f64);
            for u in begin..end {
                let d = op.degree(u);
                (s1, s2, lo, hi) = (s1 + d, s2 + d * d, lo.min(d), hi.max(d));
            }
            (s1, s2, lo, hi)
        });
        let (s1, s2, dmin, dmax) = parts.into_iter().fold(
            (0., 0., f64::INFINITY, 0f64),
            |(a, b, c, d), (e, f, g, h)| (a + e, b + f, c.min(g), d.max(h)),
        );
        let r2 = if s1 > 0. { s2 / s1 - 1. } else { 0. };
        if r2 <= 1. {
            return Err(format!(
                "error the Bethe Hessian needs a mean excess degree above 1, not {r2}: the graph is too sparse for clusters to be detectable"
            )
            .into());
        }
        let r = r2.sqrt();
        // Gershgorin: row u's eigenvalues are within r d_u of r^2 - 1 + d_u (a little margin)
        // (1 - r < 0: the least bound is the largest degree's)
        let lo = r2 - 1. + (dmin * (1. - r)).min(dmax * (1. - r));
        let hi = r2 - 1. + dmax * (1. + r);
        let margin = 1e-9 * (hi - lo).max(1.);
        (op.center, op.half) = ((lo + hi) / 2., (hi - lo) / 2. + margin);
        (op.shift, op.kappa) = (r2 - 1., -r);
        Ok(op)
    }

    /// Narrows `[center - half, center + half]` to `H`'s extreme eigenvalues, as `steps` steps of
    /// Lanczos' algorithm (from a random vector, without reorthogonalization: ghosts don't move
    /// the extremes) estimate them, widened by the last off-diagonal entry (a heuristic bound,
    /// as kernel polynomial codes use) and 1% more.
    fn tighten(&mut self, files: [&str; 2], steps: usize, seed: u64) -> Result<(), Error> {
        let n = self.g.size();
        let (mut v, mut w) = (
            Block::<f64>::new(files[0], n, 1)?,
            Block::<f64>::new(files[1], n, 1)?,
        );
        {
            let x = v.shared();
            chunks(n, self.threads, &|begin, end| {
                let mut x = x;
                for u in begin..end {
                    *x.get_mut(u) = gaussian(seed ^ 0x1a2c05, u, 0);
                }
            });
        }
        let norm = self.dots(&v, &v)[0].sqrt();
        self.scale_block(&v, 1. / norm);
        let (mut alphas, mut betas) = (vec![], vec![]);
        let mut beta = 0f64;
        for _ in 0..steps.min(n) {
            // w = M v - beta w (w holds the previous vector)
            self.step(&v, &w, 1., -beta, None);
            let alpha = self.dots(&w, &v)[0];
            self.axpy(&w, &v, -alpha);
            alphas.push(alpha);
            beta = self.dots(&w, &w)[0].sqrt();
            if beta <= 1e-12 {
                break;
            }
            betas.push(beta);
            self.scale_block(&w, 1. / beta);
            std::mem::swap(&mut v, &mut w);
        }
        let inner = &betas[..alphas.len() - 1];
        let (lo, hi) = tridiagonal_extremes(&alphas, inner);
        let last = if betas.len() == alphas.len() {
            betas.last().copied().unwrap_or(0.)
        } else {
            0.
        };
        let (lo, hi) = ((lo - last).max(-1.), (hi + last).min(1.));
        let (lo, hi) = (self.to_lambda(lo), self.to_lambda(hi));
        let margin = 0.01 * (hi - lo) + 1e-12;
        (self.center, self.half) = ((lo + hi) / 2., (hi - lo) / 2. + margin);
        Ok(())
    }

    /// `x *= factor`.
    fn scale_block(&self, x: &Block<f64>, factor: f64) {
        let b = x.b;
        let x = x.shared();
        chunks(self.g.size(), self.threads, &|begin, end| {
            let mut x = x;
            for i in begin * b..end * b {
                *x.get_mut(i) *= factor;
            }
        });
    }

    /// `y += factor x`.
    fn axpy(&self, y: &Block<f64>, x: &Block<f64>, factor: f64) {
        let b = x.b;
        let (x, y) = (x.shared(), y.shared());
        chunks(self.g.size(), self.threads, &|begin, end| {
            let mut y = y;
            for i in begin * b..end * b {
                *y.get_mut(i) += factor * *x.get(i);
            }
        });
    }

    /// `M`'s point `t` for `H`'s eigenvalue `lambda`, and back.
    fn to_t(&self, lambda: f64) -> f64 {
        (lambda - self.center) / self.half
    }
    fn to_lambda(&self, t: f64) -> f64 {
        self.center + self.half * t
    }

    /// `u`'s (weighted) degree: its row's sum.
    #[inline(always)]
    fn degree(&self, u: usize) -> f64 {
        let (offsets, labels) = (self.offsets(), self.labels());
        let (begin, end) = (offsets.get(u).index(), offsets.get(u + 1).index());
        if E::is_weighted() {
            (begin..end).map(|e| labels.get(e).weight::<f64>()).sum()
        } else {
            (end - begin) as f64
        }
    }

    fn offsets(&self) -> SharedSlice<Ix> {
        SharedSlice::<Ix>::new(self.g.offsets_ptr(), self.g.offsets_size())
    }
    fn labels(&self) -> SharedSlice<E> {
        SharedSlice::<E>::new(self.g.edge_labels_ptr(), self.g.width())
    }

    /// `next = a M cur + c prev` (`prev` is `next`'s own memory, read before it's written: rows
    /// only ever read their own), and, if given, `acc += coeff next`; all `b` wide.
    fn step<W: Weight>(
        &self,
        cur: &Block<W>,
        next: &Block<W>,
        a: f64,
        c: f64,
        acc: Option<(&Block<W>, f64)>,
    ) {
        let (n, b) = (self.g.size(), cur.b);
        let offsets = self.offsets();
        let (offsets, targets, labels) = (
            offsets.as_slice(),
            SharedSlice::<Ix>::new(self.g.neighbours_ptr(), self.g.width()),
            self.labels(),
        );
        let (targets, labels) = (targets.as_slice(), labels.as_slice());
        let scale = self.scale.as_ref().map(|s| s.as_slice());
        let (kappa, shift, slope, center, half) = (
            self.kappa / self.half,
            self.shift,
            self.slope,
            self.center,
            self.half,
        );
        let x = cur.mem.as_slice();
        let (y, acc) = (
            Rows::of(next),
            acc.map(|(block, coeff)| (Rows::of(block), coeff)),
        );
        chunks(n, self.threads, &|begin, end| {
            // rows `begin..end` are this chunk's alone
            let y = unsafe { y.rows(begin, end) };
            let mut acc = acc.map(|(r, coeff)| (unsafe { r.rows(begin, end) }, coeff));
            let mut row = vec![0f64; b];
            for u in begin..end {
                row.iter_mut().for_each(|r| *r = 0.);
                let (first, last) = (offsets[u].index(), offsets[u + 1].index());
                for e in first..last {
                    let v = targets[e].index();
                    let mut w = if E::is_weighted() {
                        labels[e].weight::<f64>()
                    } else {
                        1.
                    };
                    if let Some(scale) = scale {
                        w *= scale[v];
                    }
                    for (r, xv) in row.iter_mut().zip(&x[v * b..v * b + b]) {
                        *r += w * xv.to_f64();
                    }
                }
                let su = a * kappa * scale.map_or(1., |s| s[u]);
                // the diagonal (its degree term, the Bethe Hessian's: of unweighted graphs)
                let du = a * (shift + slope * (last - first) as f64 - center) / half;
                let at = (u - begin) * b;
                let (xs, ys) = (&x[u * b..u * b + b], &mut y[at..at + b]);
                for j in 0..b {
                    let mut value = su * row[j];
                    if du != 0. {
                        value += du * xs[j].to_f64();
                    }
                    if c != 0. {
                        value += c * ys[j].to_f64();
                    }
                    ys[j] = W::from_f64(value);
                    if let Some((acc, coeff)) = acc.as_mut() {
                        acc[at + j] = W::from_f64(acc[at + j].to_f64() + *coeff * value);
                    }
                }
            }
        });
    }

    /// Each column's `<x, y>`.
    fn dots<W: Weight>(&self, x: &Block<W>, y: &Block<W>) -> Vec<f64> {
        let b = x.b;
        let (x, y) = (x.mem.as_slice(), y.mem.as_slice());
        let parts = chunks(self.g.size(), self.threads, &|begin, end| {
            let mut sums = vec![0f64; b];
            let rows = x[begin * b..end * b]
                .chunks_exact(b)
                .zip(y[begin * b..end * b].chunks_exact(b));
            for (xr, yr) in rows {
                for ((s, xv), yv) in sums.iter_mut().zip(xr).zip(yr) {
                    *s += xv.to_f64() * yv.to_f64();
                }
            }
            sums
        });
        let mut total = vec![0f64; b];
        for part in parts {
            for (t, p) in total.iter_mut().zip(part) {
                *t += p;
            }
        }
        total
    }
}

/// The least and largest eigenvalues of the symmetric tridiagonal matrix of diagonal `a` and
/// off-diagonal `b`, by bisection on Sturm sequences.
fn tridiagonal_extremes(a: &[f64], b: &[f64]) -> (f64, f64) {
    // the number of eigenvalues below x
    let below = |x: f64| -> usize {
        let (mut count, mut q) = (0, 1f64);
        for i in 0..a.len() {
            q = a[i] - x - if i > 0 { b[i - 1] * b[i - 1] / q } else { 0. };
            if q == 0. {
                q = -1e-300;
            }
            if q < 0. {
                count += 1;
            }
        }
        count
    };
    let radius = (0..a.len())
        .map(|i| {
            a[i].abs()
                + if i > 0 { b[i - 1].abs() } else { 0. }
                + if i < b.len() { b[i].abs() } else { 0. }
        })
        .fold(0., f64::max);
    // the least x with more than `target` eigenvalues below it
    let find = |target: usize| -> f64 {
        let (mut lo, mut hi) = (-radius - 1., radius + 1.);
        for _ in 0..100 {
            let mid = (lo + hi) / 2.;
            if below(mid) > target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    };
    (find(0), find(a.len() - 1))
}

/// Jackson's damping of a degree-`m` Chebyshev expansion's `j`-th coefficient.
fn jackson(j: usize, m: usize) -> f64 {
    let a = std::f64::consts::PI / (m + 2) as f64;
    let j = j as f64;
    ((1. - j / (m + 2) as f64) * a.sin() * (j * a).cos() + (j * a).sin() * a.cos() / (m + 2) as f64)
        / a.sin()
}

/// The Jackson-damped Chebyshev coefficients, up to degree `m`, of the step `1[x <= cut]` on
/// `[-1, 1]`.
fn step_coefficients(cut: f64, m: usize) -> Vec<f64> {
    let theta = cut.clamp(-1., 1.).acos();
    let pi = std::f64::consts::PI;
    (0..=m)
        .map(|j| {
            let c = if j == 0 {
                (pi - theta) / pi
            } else {
                -2. * (j as f64 * theta).sin() / (j as f64 * pi)
            };
            c * jackson(j, m)
        })
        .collect()
}

/// The estimated number of `M`'s eigenvalues up to `t`, from the moments `mu` (`mu[j]`, the mean
/// of `r^T T_j(M) r` over random `±1` signals `r`).
fn count(mu: &[f64], t: f64) -> f64 {
    step_coefficients(t, mu.len() - 1)
        .iter()
        .zip(mu)
        .map(|(c, u)| c * u)
        .sum()
}

/// The `k` least eigenvalues' cut-off: the least `t` whose estimated eigenvalue count reaches
/// `k`.
fn cut_off(mu: &[f64], k: usize) -> f64 {
    let (mut lo, mut hi) = (-1f64, 1f64);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.;
        if count(mu, mid) >= k as f64 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// The squared distance between a row and a centroid.
#[inline(always)]
fn distance(x: impl Iterator<Item = f64>, c: &[f64]) -> f64 {
    x.zip(c).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// The nearest of `centroids` (`k × d`, row-major) to a row: its index (the least, of equals)
/// and squared distance.
#[inline(always)]
fn nearest(row: &[f64], centroids: &[f64], d: usize) -> (usize, f64) {
    let mut best = (0, f64::INFINITY);
    for (c, centroid) in centroids.chunks_exact(d).enumerate() {
        let dist = distance(row.iter().copied(), centroid);
        if dist < best.1 {
            best = (c, dist);
        }
    }
    best
}

/// k-means of `points` (`n × d`, row-major): k-means++ seeding (Arthur and Vassilvitskii), then
/// Lloyd's iterations until no point moves (at most 100), from `restarts` seeds, keeping the
/// least inertia. The centroids (`k × d`).
fn kmeans(
    points: &[f64],
    d: usize,
    k: usize,
    restarts: usize,
    seed: u64,
    threads: usize,
) -> Vec<f64> {
    let n = points.len() / d;
    let row = |i: usize| &points[i * d..i * d + d];
    let mut best = (f64::INFINITY, vec![]);
    for restart in 0..restarts.max(1) {
        let seed = mix(seed ^ mix(restart as u64 + 1));
        // k-means++
        let mut centroids = Vec::with_capacity(k * d);
        let first = (mix(seed) % n as u64) as usize;
        centroids.extend_from_slice(row(first));
        let mut closest: Vec<f64> = (0..n)
            .map(|i| distance(row(i).iter().copied(), row(first)))
            .collect();
        for c in 1..k {
            let total: f64 = closest.iter().sum();
            let pick = if total > 0. {
                let mut target = uniform(bits(seed, c, 0)) * total;
                let mut pick = n - 1;
                for (i, &w) in closest.iter().enumerate() {
                    if target < w {
                        pick = i;
                        break;
                    }
                    target -= w;
                }
                pick
            } else {
                (bits(seed, c, 1) % n as u64) as usize
            };
            centroids.extend_from_slice(row(pick));
            for (i, cl) in closest.iter_mut().enumerate() {
                *cl = cl.min(distance(row(i).iter().copied(), row(pick)));
            }
        }
        // Lloyd's
        let mut assignment = vec![usize::MAX; n];
        let mut inertia = f64::INFINITY;
        for _ in 0..100 {
            let found = chunks(n, threads, &|begin, end| {
                (begin..end)
                    .map(|i| nearest(row(i), &centroids, d))
                    .collect::<Vec<_>>()
            });
            let mut moved = false;
            inertia = 0.;
            for (i, (c, dist)) in found.into_iter().flatten().enumerate() {
                moved |= assignment[i] != c;
                assignment[i] = c;
                inertia += dist;
            }
            if !moved {
                break;
            }
            let mut sums = vec![0f64; k * d];
            let mut sizes = vec![0usize; k];
            for (i, &c) in assignment.iter().enumerate() {
                sizes[c] += 1;
                for (s, x) in sums[c * d..c * d + d].iter_mut().zip(row(i)) {
                    *s += x;
                }
            }
            for c in 0..k {
                if sizes[c] > 0 {
                    for j in 0..d {
                        centroids[c * d + j] = sums[c * d + j] / sizes[c] as f64;
                    }
                }
            }
        }
        if inertia < best.0 {
            best = (inertia, centroids);
        }
    }
    best.1
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType, W: Weight>
    SpectralClustering<'a, N, E, Ix, W>
{
    /// Clusters `g` as `options` say.
    pub fn new(g: &'a Graph<N, E, Ix>, options: SpectralOptions) -> Result<Self, Error> {
        let n = g.size();
        if n == 0 {
            return Err("error there are no nodes to cluster".into());
        }
        match (options.k, options.matrix) {
            (Some(k), _) if k == 0 || k > n => {
                return Err(format!("error k must be in 1..={n}, not {k}").into());
            }
            (None, SpectralMatrix::RegularizedLaplacian) => {
                return Err(
                    "error the regularized Laplacian needs k: only the Bethe Hessian counts clusters"
                        .into(),
                );
            }
            _ => {}
        }
        if options.matrix == SpectralMatrix::BetheHessian && options.regularization.is_some() {
            return Err("error regularization is the regularized Laplacian's only".into());
        }
        for (name, x) in [
            ("filter degree", options.filter_degree),
            ("moments", options.moments),
            ("probes", options.probes),
        ] {
            if x < 2 {
                return Err(format!("error the {name} must be at least 2, not {x}").into());
            }
        }
        let threads = g.thread_num().max(1);
        let file = |seq| g.build_cache_filename(CacheFile::Spectral, Some(seq));
        let result = (|| -> Result<_, Error> {
            let op = match options.matrix {
                SpectralMatrix::RegularizedLaplacian => {
                    Operator::laplacian(g, options.regularization, &file(0)?)?
                }
                SpectralMatrix::BetheHessian => Operator::bethe_hessian(g)?,
            };
            let mut op = op;
            op.tighten([&file(6)?, &file(7)?], 40, options.seed)?;

            // the eigenvalue counts, from the Chebyshev moments of ±1 probes: T_{j+1} = 2M T_j -
            // T_{j-1}, and mu_{2j} = 2<T_j, T_j> - mu_0, mu_{2j+1} = 2<T_{j+1}, T_j> - mu_1
            let p = options.probes;
            let (mut cur, mut other) = (
                Block::<W>::new(&file(1)?, n, p)?,
                Block::<W>::new(&file(2)?, n, p)?,
            );
            {
                let x = cur.shared();
                chunks(n, threads, &|begin, end| {
                    let mut x = x;
                    for i in begin * p..end * p {
                        let sign = if bits(options.seed ^ 0xa11ce, i / p, i % p) & 1 == 0 {
                            1.
                        } else {
                            -1.
                        };
                        *x.get_mut(i) = W::from_f64(sign);
                    }
                });
            }
            let half = options.moments.div_ceil(2);
            let mut mu = vec![0f64; 2 * half + 2];
            let mean = |v: Vec<f64>| v.iter().sum::<f64>() / p as f64;
            mu[0] = mean(op.dots(&cur, &cur));
            // T_1 = M T_0, into `other`
            op.step(&cur, &other, 1., 0., None);
            mu[1] = mean(op.dots(&other, &cur));
            std::mem::swap(&mut cur, &mut other);
            // now `cur` = T_1, `other` = T_0
            for j in 1..=half {
                mu[2 * j] = 2. * mean(op.dots(&cur, &cur)) - mu[0];
                // T_{j+1} into `other` (which held T_{j-1})
                op.step(&cur, &other, 2., -1., None);
                mu[2 * j + 1] = 2. * mean(op.dots(&other, &cur)) - mu[1];
                std::mem::swap(&mut cur, &mut other);
            }
            mu.truncate(options.moments + 1);
            drop((cur, other));
            let mut estimate = None;
            let k = match options.k {
                Some(k) => k,
                // the Bethe Hessian's negative eigenvalues: those clearly so, below 0 by three of
                // the moments' resolutions (π / moments), so that the bulk's edge, at about 0,
                // isn't counted in part
                None => {
                    let t = op.to_t(0.) - 3. * std::f64::consts::PI / options.moments as f64;
                    let counted = count(&mu, t.max(-1.));
                    estimate = Some(counted);
                    let k = counted.round().max(1.) as usize;
                    if k > options.max_k.min(n) {
                        return Err(format!(
                            "error {k} clusters estimated, more than max_k = {}: give k",
                            options.max_k
                        )
                        .into());
                    }
                    k
                }
            };
            let t_k = cut_off(&mu, k);
            let lambda_k = op.to_lambda(t_k);
            if k == 1 {
                return Ok((vec![0; n], lambda_k, estimate));
            }
            let sample = options
                .sample
                .unwrap_or_else(|| (20 * k).max((2. * k as f64 * (k as f64).ln()).ceil() as usize))
                .clamp(k, n);
            let d = options
                .features
                .unwrap_or_else(|| (4. * (sample as f64).ln()).ceil() as usize)
                .max(1);

            // features: the low-pass filter on Gaussian signals
            let m = options.filter_degree;
            let coeffs = step_coefficients(t_k, m);
            let (mut prev, mut cur) = (
                Block::<W>::new(&file(3)?, n, d)?,
                Block::<W>::new(&file(4)?, n, d)?,
            );
            let features = Block::<W>::new(&file(5)?, n, d)?;
            {
                let (mut x, mut f) = (prev.shared(), features.shared());
                chunks(n, threads, &|begin, end| {
                    let (mut x, mut f) = (x, f);
                    for i in begin * d..end * d {
                        let g = gaussian(options.seed, i / d, i % d) / (d as f64).sqrt();
                        *x.get_mut(i) = W::from_f64(g);
                        *f.get_mut(i) = W::from_f64(coeffs[0] * g);
                    }
                });
                let _ = (&mut x, &mut f);
            }
            op.step(&prev, &cur, 1., 0., Some((&features, coeffs[1])));
            for &c in &coeffs[2..] {
                // T_{j+1} into `prev`'s memory
                op.step(&cur, &prev, 2., -1., Some((&features, c)));
                std::mem::swap(&mut prev, &mut cur);
            }
            drop((prev, cur));
            // rows normalized
            {
                let mut f = features.shared();
                chunks(n, threads, &|begin, end| {
                    let mut f = f;
                    for u in begin..end {
                        let norm = (u * d..u * d + d)
                            .map(|i| f.get(i).to_f64().powi(2))
                            .sum::<f64>()
                            .sqrt();
                        if norm > 0. {
                            for i in u * d..u * d + d {
                                *f.get_mut(i) = W::from_f64(f.get(i).to_f64() / norm);
                            }
                        }
                    }
                });
                let _ = &mut f;
            }

            // k-means on a sample, then every node to its nearest centroid, refined
            let f = features.shared();
            let row =
                |u: usize| -> Vec<f64> { (u * d..u * d + d).map(|i| f.get(i).to_f64()).collect() };
            let picked: Vec<usize> = if sample >= n {
                (0..n).collect()
            } else {
                // the `sample` nodes of least hashes: a uniform sample, the same whatever the order
                let mut keyed: Vec<(u64, usize)> = chunks(n, threads, &|begin, end| {
                    (begin..end)
                        .map(|u| (bits(options.seed ^ 0x5a, u, 0), u))
                        .collect::<Vec<_>>()
                })
                .into_iter()
                .flatten()
                .collect();
                keyed.select_nth_unstable(sample - 1);
                keyed.truncate(sample);
                let mut picked: Vec<usize> = keyed.into_iter().map(|p| p.1).collect();
                picked.sort_unstable();
                picked
            };
            let points: Vec<f64> = picked.iter().flat_map(|&u| row(u)).collect();
            let mut centroids = kmeans(&points, d, k, options.restarts, options.seed, threads);
            let mut labels = vec![0usize; n];
            for iteration in 0..=options.refine {
                let found = chunks(n, threads, &|begin, end| {
                    let mut sums = vec![0f64; k * d];
                    let mut sizes = vec![0usize; k];
                    let assigned: Vec<usize> = (begin..end)
                        .map(|u| {
                            let r = row(u);
                            let c = nearest(&r, &centroids, d).0;
                            sizes[c] += 1;
                            for (s, x) in sums[c * d..c * d + d].iter_mut().zip(&r) {
                                *s += x;
                            }
                            c
                        })
                        .collect();
                    (assigned, sums, sizes)
                });
                let mut sums = vec![0f64; k * d];
                let mut sizes = vec![0usize; k];
                let mut at = 0;
                for (assigned, s, z) in found {
                    for c in assigned {
                        labels[at] = c;
                        at += 1;
                    }
                    for (a, b) in sums.iter_mut().zip(s) {
                        *a += b;
                    }
                    for (a, b) in sizes.iter_mut().zip(z) {
                        *a += b;
                    }
                }
                if iteration < options.refine {
                    for c in 0..k {
                        if sizes[c] > 0 {
                            for j in 0..d {
                                centroids[c * d + j] = sums[c * d + j] / sizes[c] as f64;
                            }
                        }
                    }
                }
            }
            drop(features);
            drop(op);
            Ok((labels, lambda_k, estimate))
        })();
        g.cleanup_cache(CacheFile::Spectral)?;
        let (labels, lambda_k, estimate) = result?;

        // clusters numbered by their least node
        let mut number = vec![usize::MAX; labels.iter().max().map_or(0, |&m| m + 1)];
        let mut count = 0;
        let out_fn = g.build_cache_filename(CacheFile::Spectral, None)?;
        let mut communities = SharedSliceMut::<Ix>::abst_mem_mut(&out_fn, n.max(1), true)?;
        for (u, c) in communities.as_mut_slice()[..n].iter_mut().zip(&labels) {
            if number[*c] == usize::MAX {
                number[*c] = count;
                count += 1;
            }
            *u = Ix::new(number[*c]);
        }
        let _ = labels.len();
        communities.flush()?;
        let modularity = g.modularity(&communities.as_slice()[..n], count)?;
        Ok(Self {
            g,
            communities,
            count,
            lambda_k,
            estimate: estimate.map(|c| (c, (2. * c.max(1.) / options.probes as f64).sqrt())),
            modularity,
            _w: std::marker::PhantomData,
        })
    }

    /// Each node's cluster, numbered `0..count` by their least node.
    pub fn communities(&self) -> &[Ix] {
        &self.communities.as_slice()[..self.g.size()]
    }

    /// The number of (nonempty) clusters: `k` at most.
    pub fn community_count(&self) -> usize {
        self.count
    }

    /// The clusters' modularity.
    pub fn partition_modularity(&self) -> f64 {
        self.modularity
    }

    /// The estimated `k`-th least eigenvalue of the matrix (the regularized normalized
    /// Laplacian, or the Bethe Hessian): the filter's cut-off.
    pub fn lambda_k(&self) -> f64 {
        self.lambda_k
    }

    /// If `k` was estimated (with the [Bethe Hessian](SpectralMatrix::BetheHessian)): the count
    /// of its clearly negative eigenvalues, unrounded, and the count's standard error,
    /// `sqrt(2k / probes)` (Hutchinson's estimator's, on a projector of rank `k`). Rounding it
    /// may miss by one or more: give `k`, if known, or more probes.
    pub fn k_estimate(&self) -> Option<(f64, f64)> {
        self.estimate
    }

    /// Removes the clusters from the cache.
    pub fn drop_cache(&mut self) -> Result<(), Error> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::Spectral, None)?;
        crate::shared_slice::remove_if_exists(out_fn)?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::graph::label::VoidLabel;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};

    /// An undirected graph of `edges` `(u, v, weight)` (each once), weighted or not.
    fn graph<E: graph::E>(
        name: &str,
        n: usize,
        edges: &[(usize, usize, f64)],
        threads: u8,
    ) -> TestGraph<VoidLabel, E, usize> {
        let dir = unique_test_dir("spectral");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let field = if E::is_weighted() { "real" } else { "pattern" };
        let mut text = format!(
            "%%MatrixMarket matrix coordinate {field} symmetric\n{n} {n} {}\n",
            edges.len()
        );
        for &(u, v, w) in edges {
            let (u, v) = (u.max(v), u.min(v));
            text += &if E::is_weighted() {
                format!("{} {} {w}\n", u + 1, v + 1)
            } else {
                format!("{} {}\n", u + 1, v + 1)
            };
        }
        std::fs::write(&path, text).unwrap();
        let id = format!("spectral_{name}_{}", unique_run_tag());
        TestGraph::new(
            GraphMemoryMap::from_file(&path, Some(id), Some(threads)).unwrap(),
            Some(dir),
        )
    }

    /// A planted partition: `k` blocks of `size` nodes, each pair linked with probability `p_in`
    /// within a block, `p_out` across; and the blocks.
    pub(crate) fn planted(
        k: usize,
        size: usize,
        p_in: f64,
        p_out: f64,
        seed: u64,
    ) -> (Vec<(usize, usize, f64)>, Vec<usize>) {
        let n = k * size;
        let mut edges = vec![];
        for u in 0..n {
            for v in u + 1..n {
                let p = if u / size == v / size { p_in } else { p_out };
                if uniform(bits(seed, u, v)) < p {
                    edges.push((u, v, 1.));
                }
            }
        }
        (edges, (0..n).map(|u| u / size).collect())
    }

    /// Normalized mutual information of two labelings (arithmetic mean normalization).
    pub(crate) fn nmi(a: &[usize], b: &[usize]) -> f64 {
        use std::collections::HashMap;
        let n = a.len() as f64;
        let (mut ca, mut cb, mut cab) = (HashMap::new(), HashMap::new(), HashMap::new());
        for (&x, &y) in a.iter().zip(b) {
            *ca.entry(x).or_insert(0.) += 1.;
            *cb.entry(y).or_insert(0.) += 1.;
            *cab.entry((x, y)).or_insert(0.) += 1.;
        }
        let h = |c: &HashMap<usize, f64>| -c.values().map(|&v| v / n * (v / n).ln()).sum::<f64>();
        let mi: f64 = cab
            .iter()
            .map(|(&(x, y), &v)| v / n * (v * n / (ca[&x] * cb[&y])).ln())
            .sum();
        let (ha, hb) = (h(&ca), h(&cb));
        if ha + hb == 0. {
            1.
        } else {
            2. * mi / (ha + hb)
        }
    }

    fn labels<Ix: graph::IndexType>(c: &[Ix]) -> Vec<usize> {
        c.iter().map(|x| x.index()).collect()
    }

    /// Planted partitions are found, the same whatever the threads, in `f64` or `f32`.
    #[test]
    fn finds_planted_partitions() {
        let (edges, truth) = planted(4, 250, 0.08, 0.004, 1);
        let mut found = vec![];
        for threads in [1u8, 4, 7] {
            let g = graph::<()>(&format!("planted_{threads}"), 1000, &edges, threads);
            let mut s =
                SpectralClustering::<_, _, _, f64>::new(&g, SpectralOptions::new(4)).unwrap();
            let got = labels(s.communities());
            assert!(
                nmi(&got, &truth) > 0.95,
                "{threads} threads: nmi {}",
                nmi(&got, &truth)
            );
            assert_eq!(s.community_count(), 4);
            let want = g.modularity(s.communities(), 4).unwrap();
            assert_eq!(s.partition_modularity(), want);
            found.push(got);
            s.drop_cache().unwrap();
            let mut s =
                SpectralClustering::<_, _, _, f32>::new(&g, SpectralOptions::new(4)).unwrap();
            assert!(nmi(&labels(s.communities()), &truth) > 0.95);
            s.drop_cache().unwrap();
        }
        assert!(found.windows(2).all(|f| f[0] == f[1]));
    }

    /// Disjoint cliques: exactly, clusters numbered by their least node.
    #[test]
    fn separates_cliques() {
        let sizes = [12, 30, 7, 20, 15];
        let mut edges = vec![];
        let mut truth = vec![];
        let mut start = 0;
        for (c, &s) in sizes.iter().enumerate() {
            for u in start..start + s {
                truth.push(c);
                for v in u + 1..start + s {
                    edges.push((u, v, 1.));
                }
            }
            start += s;
        }
        let g = graph::<()>("cliques", start, &edges, 3);
        let mut s = SpectralClustering::<_, _, _, f64>::new(&g, SpectralOptions::new(5)).unwrap();
        assert_eq!(labels(s.communities()), truth);
        s.drop_cache().unwrap();
    }

    /// Weights alone (a uniform random graph, heavy within blocks) make the clusters.
    #[test]
    fn weights_make_clusters() {
        let (mut edges, truth) = planted(3, 200, 0.03, 0.03, 2);
        for e in edges.iter_mut() {
            e.2 = if truth[e.0] == truth[e.1] { 10. } else { 0.5 };
        }
        let w = graph::<Weighted<f64>>("weights", 600, &edges, 4);
        let mut s = SpectralClustering::<_, _, _, f64>::new(&w, SpectralOptions::new(3)).unwrap();
        assert!(nmi(&labels(s.communities()), &truth) > 0.9);
        s.drop_cache().unwrap();
        let u = graph::<()>("no_weights", 600, &edges, 4);
        let mut s = SpectralClustering::<_, _, _, f64>::new(&u, SpectralOptions::new(3)).unwrap();
        assert!(nmi(&labels(s.communities()), &truth) < 0.2);
        s.drop_cache().unwrap();
    }

    #[test]
    fn bad_options_are_errors() {
        let g = graph::<()>("bad", 10, &[(0, 1, 1.)], 2);
        for k in [0, 11] {
            assert!(SpectralClustering::<_, _, _, f64>::new(&g, SpectralOptions::new(k)).is_err());
        }
        let options = SpectralOptions {
            regularization: Some(-1.),
            ..SpectralOptions::new(2)
        };
        assert!(SpectralClustering::<_, _, _, f64>::new(&g, options).is_err());
        let options = SpectralOptions {
            filter_degree: 1,
            ..SpectralOptions::new(2)
        };
        assert!(SpectralClustering::<_, _, _, f64>::new(&g, options).is_err());
    }

    /// The Bethe Hessian counts the planted blocks, and finds them.
    #[test]
    fn bethe_hessian_finds_and_counts_blocks() {
        let (edges, truth) = planted(5, 300, 0.05, 0.004, 3);
        let g = graph::<()>("bethe", 1500, &edges, 4);
        let mut s =
            SpectralClustering::<_, _, _, f64>::new(&g, SpectralOptions::bethe_hessian(5)).unwrap();
        assert!(s.lambda_k() < 0.);
        assert!(nmi(&labels(s.communities()), &truth) > 0.95);
        assert_eq!(s.k_estimate(), None);
        s.drop_cache().unwrap();
        // counted: within three standard errors
        // at a finer resolution than the defaults', as counting needs
        let counting = SpectralOptions {
            k: None,
            moments: 512,
            probes: 64,
            ..SpectralOptions::bethe_hessian(1)
        };
        let mut s = SpectralClustering::<_, _, _, f64>::new(&g, counting).unwrap();
        let (estimate, error) = s.k_estimate().unwrap();
        assert!((estimate - 5.).abs() <= 3. * error, "{estimate} ± {error}");
        s.drop_cache().unwrap();
        // no structure: one cluster (the leading eigenvector's eigenvalue, always negative)
        let (edges, _) = planted(1, 1500, 0.01, 0.01, 4);
        let g = graph::<()>("bethe_none", 1500, &edges, 4);
        let mut s = SpectralClustering::<_, _, _, f64>::new(&g, counting).unwrap();
        let (estimate, error) = s.k_estimate().unwrap();
        assert!((estimate - 1.).abs() <= 3. * error, "{estimate} ± {error}");
        s.drop_cache().unwrap();
    }

    #[test]
    fn bethe_hessian_misuse_is_an_error() {
        let (edges, _) = planted(2, 50, 0.2, 0.01, 5);
        let w = graph::<Weighted<f64>>("bethe_weighted", 100, &edges, 2);
        let err = SpectralClustering::<_, _, _, f64>::new(&w, SpectralOptions::bethe_hessian(2));
        assert!(err.err().unwrap().to_string().contains("unweighted"));
        let u = graph::<()>("laplacian_no_k", 100, &edges, 2);
        let options = SpectralOptions {
            k: None,
            ..SpectralOptions::new(2)
        };
        assert!(SpectralClustering::<_, _, _, f64>::new(&u, options).is_err());
        // a path: too sparse
        let path: Vec<_> = (0..99).map(|u| (u, u + 1, 1.)).collect();
        let p = graph::<()>("bethe_path", 100, &path, 2);
        let err = SpectralClustering::<_, _, _, f64>::new(&p, SpectralOptions::bethe_hessian(2));
        assert!(err.err().unwrap().to_string().contains("too sparse"));
    }
}
