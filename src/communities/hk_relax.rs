use crate::graph::{self, *};
use crate::utils::{f64_is_nomal, f64_to_usize_safe};

use std::collections::{HashMap, VecDeque};

/// For the computation of the *HK-Relax Algorithm* as described in ["Heat Kernel Based Community Detection"](https://doi.org/10.48550/arXiv.1403.3148) by Kloster K. and Gleich D. on [`GraphMemoryMap`] instances.
///
/// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
#[derive(Clone)]
pub struct HKRelax<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// The graph for which the community is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Diffusion depth.
    pub n: usize,
    /// Diffusion temperature.
    pub t: f64,
    /// Diffusion error (ε).
    pub eps: f64,
    /// Diffusion seed(s) (starting nodes).
    pub seed: Vec<usize>,
    /// Alogrithm parameters.
    psis: Vec<f64>,
    /// Partition's target node number --- defaults to half the graph (tolerance up to full graph).
    pub target_size: Option<usize>,
    /// Partition's target edge number --- defaults to quarter of the graph (tolerance up to half graph).
    pub target_volume: Option<usize>,
}

impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> HKRelax<'a, N, E, Ix> {
    /// Evaluates parameters for the *HK-Relax Algorithm* as described in ["Heat Kernel Based Community Detection"](https://doi.org/10.48550/arXiv.1403.3148) by Kloster K. and Gleich D.
    ///
    /// Evaluation is successful if `|V| >= 0`, `t` is normal and bigger than zero (not equal), `ε` is normal and (exclusive) between zero and one and `seed` is not empty and everyone of its entries is a valid node id, i.e. `0 <= seed[i] < |V|`.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the community is computed.
    /// * `t` --- temperature parameter.
    /// * `eps` --- ε (eps) error parameter.
    /// * `seed` --- seed nodes.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    fn evaluate_params(
        g: &GraphMemoryMap<N, E, Ix>,
        t: f64,
        eps: f64,
        seed: Vec<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = g.size();
        if node_count == 0 {
            return Err("error hk-relax invalid parameters: |V| == 0, the graph is empty".into());
        };
        if !t.is_normal() || t <= 0f64 {
            return Err(format!(
                "error hk-relax invalid parameters: t == {t} doesn't satisfy t > 0.0",
            )
            .into());
        }
        if !eps.is_normal() || eps <= 0f64 || eps >= 1f64 {
            return Err(format!(
                "error hk-relax invalid parameters: ε == {eps} doesn't satisfy 0.0 < ε 1.0",
            )
            .into());
        }
        if seed.is_empty() {
            return Err(
                "error hk-relax invalid parameters: seed_nodes.len() == 0, please provide at least one seed node".into()
            );
        }
        for (idx, seed_node) in seed.iter().enumerate() {
            if *seed_node > node_count - 1 {
                return Err(format!(
                    "error hk-relax invalid parameters: id(seed_nodes[{idx}]) == {seed_node} but max_id(v in V) == {}",
                    node_count - 1
                ).into());
            }
        }
        Ok(())
    }

    /// Given an `n` (diffusion depth) and `t` (diffusion temperature) parameters calculates the weighted
    /// sum of the errors at each individual term of the sum approximating the heat kernel
    /// diffusion vector, h, at each term, `ψ_k(t)`.
    ///
    /// [^1]: h = s + t/1 * P * s + ··· + t / n! * P^n * s.
    ///
    /// # Arguments
    ///
    /// * `n` --- diffusion depth parameter.
    /// * `t` --- diffusion temperature parameter.
    ///
    fn compute_psis(n: usize, t: f64) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut psis = vec![0f64; n + 1];
        psis[n] = 1f64;
        for i in (0..n).rev() {
            psis[i] = f64_is_nomal(
                psis[i + 1] * t / (i as f64 + 1f64) + 1f64,
                format!(
                    "{{at round: {}}} (psis[{i} + 1] * t / (i + 1) + 1)",
                    n - i + 1,
                )
                .as_str(),
            )?;
        }
        Ok(psis)
    }

    /// Given a `t` (diffusion temperature) and an `eps` (ε) parameters calculates the appropriate depth for the heat kernel diffusion.
    ///
    /// # Arguments
    ///
    /// * `t`: `f64` --- diffusion temperature parameter.
    /// * `eps`: `f64` --- diffusion ε (eps) error parameter.
    ///
    fn compute_n(t: f64, eps: f64) -> Result<usize, Box<dyn std::error::Error>> {
        let bound = f64_is_nomal(eps / 2f64, "ε/2")?;

        let n_plus_two_i32 = match i32::try_from(t.floor() as i64) {
            Ok(n) => match n.overflowing_add(1) {
                (r, false) => r,
                (_, true) => {
                    return Err(
                        format!("error computing n + 2 for hk-relax {n} + 1 overflowed",).into(),
                    );
                }
            },
            Err(e) => {
                return Err(format!(
                    "error computing n for hk-relax couldn't cast t: {t} to i32: {e}",
                )
                .into());
            }
        };

        let mut t_power_n_plus_one = f64_is_nomal(t.powi(n_plus_two_i32 - 1), "t^(n + 1)")?;

        let mut n_plus_one_fac = match (2..n_plus_two_i32).try_fold(1.0_f64, |acc, x| {
            let res = acc * (x as f64);
            if res.is_finite() { Some(res) } else { None }
        }) {
            Some(fac) => f64_is_nomal(fac, "(n + 1)!")?,
            None => {
                return Err(format!(
                    "error computing n for hk-relax overflowed trying to compute ({})!",
                    n_plus_two_i32 - 1
                )
                .into());
            }
        };

        let mut n_plus_two = n_plus_two_i32 as f64;

        while (t_power_n_plus_one * n_plus_two / n_plus_one_fac / (n_plus_two - t)) >= bound {
            t_power_n_plus_one = f64_is_nomal(t_power_n_plus_one * t, "t^(n + 1)")?;
            n_plus_one_fac = f64_is_nomal(n_plus_one_fac * n_plus_two, "(n + 1)!")?;
            n_plus_two += 1f64;
        }

        // convert the f64 value to usize subtract 2 and output n
        match f64_to_usize_safe(n_plus_two) {
            Some(n_plus_two_usize) => match n_plus_two_usize.overflowing_sub(2) {
                (n, false) => Ok(n),
                (_, true) =>
                    Err(
                    format!(
                        "error computing n for hk-relax overflowed trying to compute {n_plus_two_usize} - 2",
                    ).into())
            },
            None => Err(format!("error computing n for hk-relax can't convert {n_plus_two} to usize",).into()),
        }
    }

    /// Initializes the *HK-Relax Algorithm* as described in ["Heat Kernel Based Community Detection"](https://doi.org/10.48550/arXiv.1403.3148) by Kloster K. and Gleich D.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the community is computed.
    /// * `t` --- diffusion temperature parameter.
    /// * `eps` --- diffusion ε (eps) error parameter.
    /// * `seed` --- seed nodes' ids.
    /// * `target_size` --- partition's target node number.
    /// * `target_volume` --- partition's target edge number.
    ///
    /// [`GraphMemoryMap`]: ../../graph/struct.GraphMemoryMap.html#
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix>,
        t: f64,
        eps: f64,
        seed: Vec<usize>,
        target_size: Option<usize>,
        target_volume: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let () = Self::evaluate_params(g, t, eps, seed.clone()).map_err(
            |e| -> Box<dyn std::error::Error> {
                format!("error creating HKRelax instance: {e}").into()
            },
        )?;

        let n = Self::compute_n(t, eps)?;
        let psis = Self::compute_psis(n, t)?;
        // println!("n computed to be {}", n);

        Ok(HKRelax {
            g,
            n,
            t,
            eps,
            seed,
            psis,
            target_size,
            target_volume,
        })
    }

    #[deprecated]
    pub fn _adjust_parameters(
        &self,
        t: f64,
        eps: f64,
        seed: Vec<usize>,
        target_size: Option<usize>,
        target_volume: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let () = Self::evaluate_params(self.g, t, eps, seed.clone()).map_err(
            |e| -> Box<dyn std::error::Error> {
                format!("error creating HKRelax instance: {e}").into()
            },
        )?;

        let n = Self::compute_n(t, eps)?;
        let g = self.g;
        let psis = Self::compute_psis(n, t)?;

        Ok(HKRelax {
            g,
            n,
            t,
            eps,
            seed,
            psis,
            target_size,
            target_volume,
        })
    }

    /// Computes the *HK-Relax Algorithm* as described in ["Heat Kernel Based Community Detection"](https://doi.org/10.48550/arXiv.1403.3148) by Kloster K. and Gleich D.
    ///
    /// As a bonus we support optional community size/volume target values for control.
    ///
    pub fn compute(&self) -> Result<Community<usize>, Box<dyn std::error::Error>> {
        let n = self.n as f64;
        let threshold_pre_u_pre_j = f64_is_nomal(self.t.exp() * self.eps / n, "e^t * ε / n")?;
        // (checks weighted edges' weights: finite, non-negative)
        self.g.total_weight()?;
        // nodes' (weighted) degrees, as they're needed
        let mut degrees: HashMap<usize, f64> = HashMap::new();
        let mut degree = |u: usize| -> Result<f64, Box<dyn std::error::Error>> {
            if let Some(&d) = degrees.get(&u) {
                return Ok(d);
            }
            let d = self.g.weighted_degree(u)?;
            degrees.insert(u, d);
            Ok(d)
        };
        let mut x: HashMap<usize, f64> = HashMap::new();
        let mut r: HashMap<(usize, usize), f64> = HashMap::new();
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

        let seed_len_f64 = self.seed.clone().len() as f64;
        for seed_node in self.seed.iter() {
            if r.contains_key(&(*seed_node, 0usize)) {
                return Err(format!(
                    "error hk-relax seed node {seed_node} is present multiple times in seed array"
                )
                .into());
            }
            // r[(s, 0)] = 1. / len(seed)
            r.insert((*seed_node, 0usize), 1f64 / seed_len_f64);
            queue.push_back((*seed_node, 0));
        }

        while let Some((v, j)) = queue.pop_front() {
            let rvj = *r
                .get(&(v, j))
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    format!("error hk-relax ({v}, {j}) present in queue but not in residual",)
                        .into()
                })?;
            // x[v] += rvj && check if r[(v, j)] is normal
            x.entry(v)
                .and_modify(|x_v| {
                    *x_v = f64_is_nomal(*x_v + rvj, "x[u] + r[(v, j)]").unwrap();
                })
                .or_insert_with(|| f64_is_nomal(rvj, "r[(v, j)]").unwrap());

            // r[(v, j)] = 0
            *r.get_mut(&(v, j))
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    format!("error hk-relax r({v}, {j}) `Some` and `None`").into()
                })? = 0f64;

            //  mass = (t * rvj / (float(j) + 1.)) / len(G[v]) /* calculation validity checked when poppped from queue */
            // (weighted: each neighbour u gets the share w(v, u) / d(v) of it)
            let deg_v = degree(v)?;
            if deg_v == 0. {
                // nothing to push to (no edges, or all weighing nothing)
                continue;
            }
            let unit_mass = self.t * rvj / (j as f64 + 1f64) / deg_v;

            for (u, label) in self.g.weighted_neighbours(v)? {
                let w = label.weight::<f64>();
                if w == 0. {
                    // carries nothing (and may lead to a node of degree 0, whose threshold
                    // would be 0)
                    continue;
                }
                let mass = unit_mass * w;
                let next = (u, j + 1);
                if j + 1 == self.n {
                    // x[u] += rvj / len(G[v])
                    let share = rvj * w / deg_v;
                    x.entry(u).and_modify(|x_u| *x_u += share).or_insert(share);
                    continue;
                }

                // if next not in r: r[next] = 0.
                let r_next = r.entry(next).or_insert(0_f64);

                // thresh = math.exp(t) * eps * len(G[u]) / (N * psis[j + 1])
                let deg_u = degree(u)?;
                let threshold = f64_is_nomal(
                    threshold_pre_u_pre_j * deg_u / self.psis[j + 1],
                    "e^t * ε * deg_u / (n * psis[j + 1])",
                )?;

                // if r[next] < thresh and r[next] + mass >= thresh:
                if *r_next < threshold && *r_next + mass >= threshold {
                    queue.push_back(next);
                }
                // r[next] = r[next] + mass
                *r_next += mass;
            }
        }

        let mut h: Vec<(usize, f64)> = Vec::with_capacity(x.len());
        for (&v, &x_v) in &x {
            h.push((v, x_v / degree(v)?));
        }

        match self
            .g
            .sweep_cut(h.as_mut(), self.target_size, self.target_volume)
        {
            Ok(c) => {
                // println!(
                //     "best community hkrelax {{\n\tsize: {}\n\tvolume/width: {}\n\tconductance: {}\n}}",
                //     c.size, c.width, c.conductance
                // );
                Ok(c)
            }
            Err(e) => Err(format!("error sweep cut: {e}").into()),
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::graph::test_utils::{TestGraph, unique_run_tag, unique_test_dir};

    /// An undirected graph of `edges` `(u, v, value)` (each listed once), as a symmetric `real`
    /// MatrixMarket file, its values read as `L`'s weights (or ignored, unweighted).
    pub(crate) fn undirected<L: graph::E>(
        name: &str,
        n: usize,
        edges: &[(usize, usize, &str)],
    ) -> TestGraph<(), L, usize, Undirected> {
        let dir = unique_test_dir("hk_weights");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.mtx"));
        let mut text = format!(
            "%%MatrixMarket matrix coordinate real symmetric\n{n} {n} {}\n",
            edges.len()
        );
        for (u, v, w) in edges {
            // (lower triangle first, as symmetric files list entries)
            text += &format!("{} {} {w}\n", u.max(v) + 1, u.min(v) + 1);
        }
        std::fs::write(&path, text).unwrap();
        let g = Graph::<(), L, usize>::from_file(
            &path,
            Some(format!("hk_{name}_{}", unique_run_tag())),
            Some(2),
        )
        .unwrap();
        TestGraph::new(g, Some(dir))
    }

    /// Two 5-cliques (0-4, 5-9) bridged by 4-5, every edge valued `w`, and `extra` edges.
    pub(crate) fn cliques<'a>(
        w: &'a str,
        extra: &[(usize, usize, &'a str)],
    ) -> Vec<(usize, usize, &'a str)> {
        let mut edges = vec![(4, 5, w)];
        for base in [0, 5] {
            for u in 0..5 {
                for v in u + 1..5 {
                    edges.push((base + u, base + v, w));
                }
            }
        }
        edges.extend_from_slice(extra);
        edges
    }

    fn sorted_nodes(c: &Community<usize>) -> Vec<usize> {
        let mut nodes: Vec<usize> = c.nodes.iter().map(|(u, _)| *u).collect();
        nodes.sort_unstable();
        nodes
    }

    /// The sweep cut weighs cuts and volumes: path 0-1-2-3 weighing 5, 1, 5 is best cut in the
    /// middle, at conductance 1 / 11 (1 / 3 unweighted), its volume the weights' 11.
    #[test]
    fn sweep_cut_weighs_cuts_and_volumes() {
        let edges = [(0, 1, "5"), (1, 2, "1"), (2, 3, "5")];
        let weighted = undirected::<Weighted<f64>>("sweep_w", 4, &edges);
        let plain = undirected::<()>("sweep_plain", 4, &edges);
        let diffusion = || vec![(0, 1.0), (1, 0.5), (2, 0.25), (3, 0.125)];
        let c = weighted.sweep_cut(&mut diffusion(), None, None).unwrap();
        assert_eq!(sorted_nodes(&c), vec![0, 1]);
        assert_eq!((c.size, c.width, c.volume), (2, 3, 11.0));
        assert!(
            (c.conductance - 1. / 11.).abs() < 1e-15,
            "{}",
            c.conductance
        );
        let c = plain.sweep_cut(&mut diffusion(), None, None).unwrap();
        assert_eq!(sorted_nodes(&c), vec![0, 1]);
        assert_eq!((c.size, c.width, c.volume), (2, 3, 3.0));
        assert!((c.conductance - 1. / 3.).abs() < 1e-15, "{}", c.conductance);
    }

    /// Weights all of `4` (a power of two: scaling by it is exact) find exactly what unit ones
    /// do, their volume four times the arcs'.
    #[test]
    fn hk_relax_uniform_weights_match_unweighted() {
        let weighted = undirected::<Weighted<f64>>("hk_uniform", 10, &cliques("4", &[]));
        let plain = undirected::<()>("hk_plain", 10, &cliques("4", &[]));
        for seed in [0, 4, 7] {
            let want = HKRelax::new(&plain, 3.0, 1e-4, vec![seed], None, None)
                .unwrap()
                .compute()
                .unwrap();
            let got = HKRelax::new(&weighted, 3.0, 1e-4, vec![seed], None, None)
                .unwrap()
                .compute()
                .unwrap();
            assert_eq!(sorted_nodes(&got), sorted_nodes(&want), "seed {seed}");
            assert_eq!(got.conductance, want.conductance, "seed {seed}");
            assert_eq!(got.width, want.width, "seed {seed}");
            assert_eq!(got.volume, 4. * want.volume, "seed {seed}");
            // the seed's clique
            assert_eq!(sorted_nodes(&want).len(), 5, "seed {seed}");
        }
    }

    /// Edges weighing nothing are as good as absent: no mass crosses them, and cuts and volumes
    /// don't count them.
    #[test]
    fn hk_relax_zero_weight_edges_are_absent() {
        let with = cliques("1", &[(0, 9, "0"), (1, 8, "0"), (2, 7, "0")]);
        let with = undirected::<Weighted<f64>>("hk_zero", 10, &with);
        let without = undirected::<Weighted<f64>>("hk_nonzero", 10, &cliques("1", &[]));
        for seed in [0, 2, 9] {
            let a = HKRelax::new(&with, 3.0, 1e-4, vec![seed], None, None)
                .unwrap()
                .compute()
                .unwrap();
            let b = HKRelax::new(&without, 3.0, 1e-4, vec![seed], None, None)
                .unwrap()
                .compute()
                .unwrap();
            assert_eq!(sorted_nodes(&a), sorted_nodes(&b), "seed {seed}");
            assert_eq!(a.conductance, b.conductance, "seed {seed}");
            assert_eq!(a.volume, b.volume, "seed {seed}");
        }
    }

    #[test]
    fn hk_relax_rejects_bad_weights() {
        for (name, w) in [("hk_negative", "-1"), ("hk_nan", "nan")] {
            let g = undirected::<Weighted<f64>>(name, 10, &cliques("1", &[(0, 9, w)]));
            let err = HKRelax::new(&g, 3.0, 1e-4, vec![0], None, None)
                .unwrap()
                .compute()
                .err()
                .unwrap()
                .to_string();
            assert!(err.contains("finite and non-negative"), "{name}: {err}");
        }
    }
}
