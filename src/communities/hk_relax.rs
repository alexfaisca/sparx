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
            let v_n = self.g.neighbours(v)?;
            let deg_v = v_n.remaining_neighbours() as f64;

            let mass = self.t * rvj / (j as f64 + 1f64) / deg_v;

            for u in v_n {
                let next = (u, j + 1);
                if j + 1 == self.n {
                    // x[u] += rvj / len(G[v])
                    x.entry(u)
                        .and_modify(|x_u| {
                            *x_u = f64_is_nomal(*x_u + rvj / deg_v, "x[u] + (r[(v, j)] / deg_v)")
                                .unwrap();
                        })
                        .or_insert_with(|| f64_is_nomal(rvj / deg_v, "r[(v, j)] / deg_v").unwrap());
                    continue;
                }

                // if next not in r: r[next] = 0.
                let r_next = r.entry(next).or_insert(0_f64);

                // thresh = math.exp(t) * eps * len(G[u]) / (N * psis[j + 1])
                let deg_u = self.g.node_degree(u) as f64;
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

        let mut h: Vec<(usize, f64)> = x
            .keys()
            .map(|v| (*v, x.get(v).unwrap() / self.g.node_degree(*v) as f64))
            .collect::<Vec<(usize, f64)>>();

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
