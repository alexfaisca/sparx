use crate::graph::{self, *};
use crate::utils::{f64_is_nomal, f64_to_usize_safe};

use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Poisson};
use std::collections::HashMap;

/// For the computation of the *ApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O. on [`GraphMemoryMap`] instances.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[derive(Clone)]
pub struct ApproxDirHKPR<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which the community is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Diffusion temperature.
    pub t: f64,
    /// Diffusion error (ε).
    pub eps: f64,
    /// Diffusion seed (starting node).
    pub seed: usize,
    /// Partition's target conductance.
    pub target_conductance: f64,
    /// Partition's target node number --- defaults to half the graph (tolerance up to full graph).
    pub target_size: usize,
    /// Partition's target edge number --- defaults to quarter of the graph (tolerance up to half graph).
    pub target_vol: usize,
    /// The random walks' and step counts' seed (see [`with_random_seed`](Self::with_random_seed)):
    /// the same seed, the same community.
    pub random_seed: u64,
}

/// Type of limiter to be used for the number of steps to take for each random walk.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum ApproxDirichletHeatKernelK {
    /// * Note: too stringent of a limitation -- results will be no good.
    None,
    /// * Note: optimization described in SolverApproxDirHKPR's algorithm.
    Mean,
    /// * Note: no limit --- infeasible for eps < 0.005 in large graphs.
    Unlim,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ApproxDirHKPR<'a, N, E, Ix> {
    /// Evaluates parameters for the *ApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O.
    ///
    /// Evaluation is successful if `|V| >= 0`, `t` is normal and bigger than zero (not equal), `ε` is normal and (exclusive) between zero and one, `target_conductance` is normal and (exclusive) between zero and one, and `seed` is a valid node id, i.e. `0 <= seed < |V|`.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the community is computed.
    /// * `seed_node` --- seed node.
    /// * `eps` --- ε (eps) error parameter.
    /// * `target_conductance` --- target conductance parameter.
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or `Err(_)` if not.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    fn evaluate_params(
        g: &GraphMemoryMap<N, E, Ix>,
        seed_node: usize,
        eps: f64,
        target_conductance: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = g.size();
        if node_count == 0 {
            return Err(
                "error approx-dirichlet-hkpr invalid parameters: actual |V| == 0, the graph is empty".into(),
            );
        };
        if !eps.is_normal() || eps <= 0f64 || eps >= 1f64 {
            return Err(format!(
                "error approx-dirichlet-hkpr  invalid parameters: ε == {eps} doesn't satisfy 0.0 < ε 1.0",
            )
            .into());
        }
        if !target_conductance.is_normal()
            || target_conductance <= 0f64
            || target_conductance >= 1f64
        {
            return Err(format!(
                "error approx-dirichlet-hkpr invalid parameters: target_conductance == {} doesn't satisfy 0.0 < target_conductance < 1.0",
                target_conductance
            ).into());
        }
        if seed_node > node_count - 1 {
            return Err(format!(
                "error approx-dirichlet-hkpr invalid parameters: id(seed_nodes) == {seed_node} but max_id(v in V) == {}",
                node_count - 1
            )
            .into());
        }
        Ok(())
    }

    /// Initializes the *ApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the community is computed.
    /// * `eps` --- ε (eps) error parameter.
    /// * `seed` --- seed node.
    /// * `target_size` --- partition's target node number.
    /// * `target_volume` --- partition's target edge number.
    /// * `target_conductance` --- partition's target conductance.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix>,
        eps: f64,
        seed: usize,
        target_size: usize,
        target_vol: usize,
        target_conductance: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let () = Self::evaluate_params(g, seed, eps, target_conductance)?;
        let t_formula = "(1. / target_conductance) * ln((2. * sqrt(target_vol)) / (1. - ε) + 2. * ε * target_size)";
        let t = f64_is_nomal(
            (1. / target_conductance)
                * f64::ln(
                    (2. * f64::sqrt(target_vol as f64)) / (1. - eps)
                        + 2. * eps * target_size as f64,
                ),
            t_formula,
        )?;
        Ok(ApproxDirHKPR {
            random_seed: 0x5eed,
            g,
            t,
            eps,
            seed,
            target_size,
            target_vol,
            target_conductance,
        })
    }

    /// The same, sampling random walks and their lengths from `seed` (by default a fixed one:
    /// a computation's community is the same every run).
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    fn random_sample_poisson(
        lambda: f64,
        n: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<OrderedFloat<f64>>, Box<dyn std::error::Error>> {
        let poisson = match Poisson::new(lambda) {
            Ok(dist) => dist,
            Err(e) => {
                return Err(format!(
                    "error approx-dirchlet-hkpr couldn't sample poission distribution: {e}",
                )
                .into());
            }
        };
        Ok(poisson
            .sample_iter(&mut *rng)
            .take(n)
            .map(OrderedFloat)
            .collect())
    }

    /// The neighbour of `u` a uniform draw `r` (in `[0, 1)`) picks: uniformly among its arcs, or
    /// for weighted arcs in proportion to their weights; `None` if `u` has no arcs (or they all
    /// weigh nothing). Linear in `u`'s degree, as was the uniform pick.
    fn pick_neighbour(
        &self,
        u: usize,
        r: f64,
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        let mut arcs = self.g.weighted_neighbours(u)?;
        if !E::is_weighted() {
            let deg = arcs.len();
            if deg == 0 {
                return Ok(None);
            }
            let i = ((r * deg as f64) as usize).min(deg - 1);
            return Ok(arcs.nth(i).map(|(v, _)| v));
        }
        let total = self.g.weighted_degree(u)?;
        if total.is_nan() || total <= 0. {
            return Ok(None);
        }
        // the arc whose share of `[0, total)` holds `r * total` (the last weighing anything, if
        // rounding overshoots)
        let target = r * total;
        let (mut at, mut last) = (0f64, None);
        for (v, label) in arcs {
            let w = label.weight::<f64>();
            if w > 0. {
                at += w;
                last = Some(v);
                if target < at {
                    return Ok(last);
                }
            }
        }
        Ok(last)
    }

    fn random_walk_seed(
        &self,
        k: usize,
        seed_node: usize,
        rng: &mut StdRng,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut curr_node = seed_node;
        for _ in 0..k {
            match self.pick_neighbour(curr_node, rng.random())? {
                Some(v) => curr_node = v,
                // nowhere to go
                None => return Ok(curr_node),
            }
        }
        Ok(curr_node)
    }

    /// Computes the *ApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O. with user controlled optimization level.
    ///
    /// # Arguments
    ///
    /// * `big_k` --- described the type of optimization to be used in random walk sampling.
    ///
    #[allow(clippy::unreachable)]
    #[deprecated]
    fn compute_specify_k(
        &self,
        big_k: ApproxDirichletHeatKernelK,
    ) -> Result<Community<usize>, Box<dyn std::error::Error>> {
        let node_count = match self.g.size() {
            0 => return Err("error approx-dirchlet-hkpr |V| == 0".into()),
            i => i as f64,
        };
        // (checks weighted edges' weights: finite, non-negative)
        self.g.total_weight()?;

        let r = f64_is_nomal(
            (16. / self.eps.powi(2)) * f64::ln(node_count),
            "(16.0 / ε²) * ln(|V|)",
        )?;
        let one_over_r = f64_is_nomal(1. / r, "1. / r")?;

        let k = match big_k {
            ApproxDirichletHeatKernelK::None => f64_is_nomal(
                2. * f64::ln(1. / self.eps) / (f64::ln(f64::ln(1. / self.eps))),
                "2. * ln(1. / ε) / ln(ln(1. / ε))",
            )?,
            // optimization of SolverApproxDirHKPR
            ApproxDirichletHeatKernelK::Mean => f64_is_nomal(2. * self.t, "2. * t")?,
            ApproxDirichletHeatKernelK::Unlim => f64::INFINITY,
            #[expect(unreachable_patterns)]
            _ => {
                return Err(format!("error approx-dirchlet-hkpr unknown K {:?}", big_k).into());
            }
        };
        let k = OrderedFloat(k);

        let num_samples: usize = match f64_to_usize_safe(r) {
            Some(s) => s,
            None => {
                return Err(
                    format!("error approx-dirchlet-hkpr couldn't cast {r} to usize").into(),
                );
            }
        };
        let mut rng = StdRng::seed_from_u64(self.random_seed);
        let steps: Vec<OrderedFloat<f64>> =
            Self::random_sample_poisson(self.t, num_samples, &mut rng)?;
        let mut aprox_hkpr_samples: HashMap<usize, f64> = HashMap::new();

        for little_k in steps {
            let OrderedFloat(little_k) = std::cmp::min(little_k, k);
            let little_k_usize = match f64_to_usize_safe(little_k) {
                Some(val) => val,
                None => {
                    return Err(format!(
                        "error approx-dirchlet-hkpr couldn't cast {little_k} to usize"
                    )
                    .into());
                }
            };
            let v = self.random_walk_seed(little_k_usize, self.seed, &mut rng)?;
            match aprox_hkpr_samples.get_mut(&v) {
                Some(v) => *v += one_over_r,
                None => {
                    aprox_hkpr_samples.insert(v, one_over_r);
                }
            };
        }

        // FIXME: Never normalized
        let mut p: Vec<(usize, f64)> = aprox_hkpr_samples
            .keys()
            .map(|u| {
                (
                    *u,
                    *aprox_hkpr_samples.get(u).unwrap() / self.g.weighted_degree(*u).unwrap_or(0.),
                )
            })
            .collect::<Vec<(usize, f64)>>();

        match self
            .g
            .sweep_cut(p.as_mut(), Some(self.target_size), Some(self.target_vol))
        {
            Ok(c) => Ok(c),
            Err(e) => Err(format!("error approx-dirichlet-hkpr performing sweepcut: {e}").into()),
        }
    }

    /// Computes the *SolverApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O. with the therein described optimizations.
    ///
    pub fn compute(&self) -> Result<Community<usize>, Box<dyn std::error::Error>> {
        let node_count = match self.g.size() {
            0 => return Err("error approx-dirchlet-hkpr |V| == 0".into()),
            i => i as f64,
        };
        // (checks weighted edges' weights: finite, non-negative)
        self.g.total_weight()?;

        let r = f64_is_nomal(
            (16. / self.eps.powi(2)) * f64::ln(node_count),
            "(16.0 / ε²) * ln(|V|)",
        )?;
        let one_over_r = f64_is_nomal(1. / r, "1. / r")?;

        let k = f64_is_nomal(2. * self.t, "2. * t")?;
        let k = OrderedFloat(k);

        let num_samples: usize = match f64_to_usize_safe(r) {
            Some(s) => s,
            None => {
                return Err(
                    format!("error approx-dirchlet-hkpr couldn't cast {r} to usize").into(),
                );
            }
        };
        let mut rng = StdRng::seed_from_u64(self.random_seed);
        let steps: Vec<OrderedFloat<f64>> =
            Self::random_sample_poisson(self.t, num_samples, &mut rng)?;
        let mut aprox_hkpr_samples: HashMap<usize, f64> = HashMap::new();

        for little_k in steps {
            let OrderedFloat(little_k) = std::cmp::min(little_k, k);
            let little_k_usize = match f64_to_usize_safe(little_k) {
                Some(val) => val,
                None => {
                    return Err(format!(
                        "error approx-dirchlet-hkpr couldn't cast {little_k} to usize"
                    )
                    .into());
                }
            };
            let v = self.random_walk_seed(little_k_usize, self.seed, &mut rng)?;
            match aprox_hkpr_samples.get_mut(&v) {
                Some(v) => *v += one_over_r,
                None => {
                    aprox_hkpr_samples.insert(v, one_over_r);
                }
            };
        }

        // FIXME: Never normalized
        let mut p: Vec<(usize, f64)> = aprox_hkpr_samples
            .keys()
            .map(|u| {
                (
                    *u,
                    *aprox_hkpr_samples.get(u).unwrap() / self.g.weighted_degree(*u).unwrap_or(0.),
                )
            })
            .collect::<Vec<(usize, f64)>>();

        match self
            .g
            .sweep_cut(p.as_mut(), Some(self.target_size), Some(self.target_vol))
        {
            Ok(c) => Ok(c),
            Err(e) => Err(format!("error approx-dirichlet-hkpr performing sweep cut: {e}").into()),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::communities::hk_relax::test::undirected;

    /// Uniform draws pick each of a node's neighbours alike (the pick once doubled its index, so
    /// that from degree 4 on later neighbours were never picked), and weighted ones in proportion
    /// to their weights, never one weighing nothing. Nodes with nowhere to go pick nothing.
    #[test]
    fn neighbours_are_picked_in_proportion() {
        // node 0: neighbours 1-4; node 5: 6 (1), 7 (0), 8 (3); node 9: 10 (0); 11 alone
        let edges = [
            (0, 1, "1"),
            (0, 2, "1"),
            (0, 3, "1"),
            (0, 4, "1"),
            (5, 6, "1"),
            (5, 7, "0"),
            (5, 8, "3"),
            (9, 10, "0"),
        ];
        let plain = undirected::<()>("pick_plain", 12, &edges);
        let weighted = undirected::<Weighted<f64>>("pick_w", 12, &edges);
        fn pick<L: graph::E>(g: &GraphMemoryMap<(), L, usize>, u: usize, r: f64) -> Option<usize> {
            ApproxDirHKPR::new(g, 0.5, 0, 2, 4, 0.5)
                .unwrap()
                .pick_neighbour(u, r)
                .unwrap()
        }
        let draws = [0.0, 0.26, 0.51, 0.76, 0.999_999];
        let picked: Vec<_> = draws.iter().map(|&r| pick(&plain, 0, r)).collect();
        assert_eq!(picked, [1, 2, 3, 4, 4].map(Some));
        // unweighted, node 5's three neighbours alike
        let picked: Vec<_> = [0.0, 0.34, 0.67]
            .iter()
            .map(|&r| pick(&plain, 5, r))
            .collect();
        assert_eq!(picked, [6, 7, 8].map(Some));
        // weighted: 6 for the first quarter, 8 for the rest, 7 never
        let picked: Vec<_> = [0.0, 0.24, 0.25, 0.5, 0.999_999]
            .iter()
            .map(|&r| pick(&weighted, 5, r))
            .collect();
        assert_eq!(picked, [6, 6, 8, 8, 8].map(Some));
        assert_eq!(pick(&weighted, 9, 0.5), None);
        assert_eq!(pick(&plain, 9, 0.5), Some(10));
        assert_eq!(pick(&plain, 11, 0.5), None);
        assert_eq!(pick(&weighted, 11, 0.5), None);
    }

    /// Weights that aren't finite and non-negative are refused.
    #[test]
    fn approx_dirichlet_hkpr_rejects_bad_weights() {
        let g = undirected::<Weighted<f64>>("adh_negative", 3, &[(0, 1, "1"), (1, 2, "-2")]);
        let err = ApproxDirHKPR::new(&g, 0.5, 0, 2, 4, 0.5)
            .unwrap()
            .compute()
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("finite and non-negative"), "{err}");
    }
}
