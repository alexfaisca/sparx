#!/usr/bin/env bash
set -euo pipefail

# dataset paths are relative to the repository's root
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
shopt -s nullglob

for ds in ./datasets/graphs/graph_3_10.lz4; do
	[[ -f "$ds" ]] || continue
	for t in {3..10}; do
		echo ">> $(basename "$ds")  -t $t"
		GLIBC_TUNABLES=glibc.malloc.arena_max=1 cargo cache --tool=massif-pages --dataset="$ds" -t "$t"
	done
	for t in {0..2}; do
		echo ">> $(basename "$ds")  -t $t"
		GLIBC_TUNABLES=glibc.malloc.arena_max=1 cargo cache --tool=massif --dataset="$ds" -t "$t"
	done
done
