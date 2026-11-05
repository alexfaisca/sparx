#!/usr/bin/env bash
# Regenerates the small ggcat graphs the test suite uses (`random_graph_<i>_<k>.lz4`): the
# de Bruijn graphs (k = 5 and k = 10) of 9 seeded, genome-like random DNA inputs of growing size.
#
# Requires `ggcat` (`cargo install --git https://github.com/algbio/ggcat --locked ggcat_cmdline`)
# and `python3`. Run from anywhere; outputs land next to this script.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

for i in $(seq 1 9); do
	input="$work/random_$i.fasta"
	# genome-like input, seeded by `i`: a random backbone with a repeated segment (cycles), and
	# `i` copies of it with ~1% point mutations (bubbles), so both k = 5 and k = 10 graphs branch
	python3 - "$i" "$input" <<'PY'
import random, sys
i, path = int(sys.argv[1]), sys.argv[2]
rng = random.Random(i)
base = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
repeat = base(40)
backbone = "".join(base(600) + repeat for _ in range(4)) + base(600)
with open(path, "w") as f:
    f.write(f">backbone\n{backbone}\n")
    for s in range(i):
        variant = "".join(rng.choice("ACGT") if rng.random() < 0.01 else c for c in backbone)
        f.write(f">variant{s}\n{variant}\n")
PY
	for k in 5 10; do
		out="$here/random_graph_${i}_${k}.lz4"
		ggcat build -k "$k" -j 4 -c -e "$input" --output-file "$out" >/dev/null 2>&1
		# ggcat's colors and stats by-products aren't needed
		rm -f "${out%.lz4}.stats.log" "${out%.lz4}.colors.dat"
	done
done
ls -la "$here"
