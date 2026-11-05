#!/usr/bin/env bash
set -euo pipefail

# dataset paths are relative to the repository's root
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/graph_1_10.lz4 -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/graph_1_10.lz4 -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/graph_1_10.lz4 -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/graph_1_10.lz4 -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/graph_1_10.lz4 -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/graph_1_10.lz4 -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/graph_1_10.lz4 -e 2

cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/graph_3_10.lz4 -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/graph_3_10.lz4 -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/graph_3_10.lz4 -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/graph_3_10.lz4 -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/graph_3_10.lz4 -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/graph_3_10.lz4 -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/graph_3_10.lz4 -e 2

cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/graph_9_15.lz4 -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/graph_9_15.lz4 -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/graph_9_15.lz4 -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/graph_9_15.lz4 -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/graph_9_15.lz4 -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/graph_9_15.lz4 -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/graph_9_15.lz4 -e 2

cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/kmer_V2a.mtx -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/kmer_V2a.mtx -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/kmer_V2a.mtx -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/kmer_V2a.mtx -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/kmer_V2a.mtx -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/kmer_V2a.mtx -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/kmer_V2a.mtx -e 2

cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/kmer_A2a.mtx -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/kmer_A2a.mtx -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/kmer_A2a.mtx -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/kmer_A2a.mtx -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/kmer_A2a.mtx -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/kmer_A2a.mtx -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/kmer_A2a.mtx -e 2

cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/kmer_V1r.mtx -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/kmer_V1r.mtx -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/kmer_V1r.mtx -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/kmer_V1r.mtx -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/kmer_V1r.mtx -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/kmer_V1r.mtx -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/kmer_V1r.mtx -e 2

cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/graph_1_5.lz4 -e 2
cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/graph_4_10.lz4 -e 2
cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/graph_9_10.lz4 -e 2
cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/synth_dna_k11_edges33_5M.mtx -e 2
cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/synth_dna_k10_edges8_4M.mtx -e 2
cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/synth_gen_dbg_dna_k7_edges130k.mtx -e 2
cargo run --features bench --release -- -t64 -v -mf ./datasets/graphs/synth_gen_dbg_protein20_k2_edges16k.mtx -e 2

cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/graph_1_5.lz4 -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/graph_4_10.lz4 -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/graph_9_10.lz4 -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/synth_dna_k11_edges33_5M.mtx -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/synth_dna_k10_edges8_4M.mtx -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/synth_gen_dbg_dna_k7_edges130k.mtx -e 2
cargo run --features bench --release -- -t32 -v -mf ./datasets/graphs/synth_gen_dbg_protein20_k2_edges16k.mtx -e 2

cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/graph_1_5.lz4 -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/graph_4_10.lz4 -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/graph_9_10.lz4 -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/synth_dna_k11_edges33_5M.mtx -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/synth_dna_k10_edges8_4M.mtx -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/synth_gen_dbg_dna_k7_edges130k.mtx -e 2
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/synth_gen_dbg_protein20_k2_edges16k.mtx -e 2

cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/graph_1_5.lz4 -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/graph_4_10.lz4 -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/graph_9_10.lz4 -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/synth_dna_k11_edges33_5M.mtx -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/synth_dna_k10_edges8_4M.mtx -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/synth_gen_dbg_dna_k7_edges130k.mtx -e 2
cargo run --features bench --release -- -t8 -v -mf ./datasets/graphs/synth_gen_dbg_protein20_k2_edges16k.mtx -e 2

cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/graph_1_5.lz4 -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/graph_4_10.lz4 -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/graph_9_10.lz4 -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/synth_dna_k11_edges33_5M.mtx -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/synth_dna_k10_edges8_4M.mtx -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/synth_gen_dbg_dna_k7_edges130k.mtx -e 2
cargo run --features bench --release -- -t4 -v -mf ./datasets/graphs/synth_gen_dbg_protein20_k2_edges16k.mtx -e 2

cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/graph_1_5.lz4 -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/graph_4_10.lz4 -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/graph_9_10.lz4 -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/synth_dna_k11_edges33_5M.mtx -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/synth_dna_k10_edges8_4M.mtx -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/synth_gen_dbg_dna_k7_edges130k.mtx -e 2
cargo run --features bench --release -- -t2 -v -mf ./datasets/graphs/synth_gen_dbg_protein20_k2_edges16k.mtx -e 2

cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/graph_1_5.lz4 -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/graph_4_10.lz4 -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/graph_9_10.lz4 -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/synth_dna_k11_edges33_5M.mtx -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/synth_dna_k10_edges8_4M.mtx -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/synth_gen_dbg_dna_k7_edges130k.mtx -e 2
cargo run --features bench --release -- -t1 -v -mf ./datasets/graphs/synth_gen_dbg_protein20_k2_edges16k.mtx -e 2
