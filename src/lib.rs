// without an input format, graphs can't be loaded at all
#[cfg(not(any(feature = "mtx", feature = "ggcat", feature = "nodes_edges")))]
compile_error!(
    "sparx needs at least one input format feature: `mtx`, `ggcat` or `nodes_edges` (all are on by default)"
);

pub mod shared_slice;

#[cfg(any(test, feature = "bench"))]
pub mod test_common;

#[cfg(test)]
mod cross_ix_sweep_test;

#[cfg(test)]
mod directed_algorithms_test;

pub mod centralities;
pub mod communities;
pub mod components;
pub mod graph;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod k_core;
pub mod k_truss;
pub mod trails;
pub mod utils;
