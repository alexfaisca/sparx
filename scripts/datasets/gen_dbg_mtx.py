#!/usr/bin/env python3
import argparse, gzip, sys

def emit_dbg_mtx(sigma: int, k: int, fh):
    if k < 1:
        raise SystemExit("k must be >= 1 (nodes are k-mers).")
    N = sigma ** k            # nodes
    M = sigma ** (k + 1)      # directed edges (each (k+1)-mer)

    print("%%MatrixMarket matrix coordinate pattern symmetric", file=fh)
    print(f"% de Bruijn graph: sigma={sigma}, k={k}; nodes={N}, edges={M}", file=fh)
    print(f"{N} {N} {M}", file=fh)

    base = sigma ** (k - 1)   # for rolling the suffix
    # Node indices are 1-based in Matrix Market
    for u in range(N):
        tail = u % base
        base_tail = tail * sigma
        up1 = u + 1
        for c in range(sigma):
            v = base_tail + c + 1
            fh.write(f"{up1} {v}\n")

def main():
    ap = argparse.ArgumentParser(description="Generate a de Bruijn graph in Matrix Market (.mtx).")
    ap.add_argument("--alphabet", choices=["dna","protein20","protein23"], default=None,
                    help="dna=4, protein20=20 amino acids, protein23=20 + {B,Z,X}")
    ap.add_argument("--sigma", type=int, default=None, help="Override alphabet size (integer).")
    ap.add_argument("--k", type=int, required=True, help="k-mer length for nodes (k>=1).")
    ap.add_argument("--out", type=str, required=True, help="Output path (.mtx or .mtx.gz). Use '-' for stdout.")
    args = ap.parse_args()

    # resolve sigma
    preset = {"dna":4, "protein20":20, "protein23":23}
    sigma = args.sigma if args.sigma is not None else preset.get(args.alphabet)
    if sigma is None:
        raise SystemExit("Provide --alphabet or --sigma.")

    # open output (gzip if .gz)
    if args.out == "-":
        fh = sys.stdout
        close = lambda: None
    elif args.out.endswith(".gz"):
        f = gzip.open(args.out, "wt", newline="\n")
        fh, close = f, f.close
    else:
        f = open(args.out, "w", newline="\n")
        fh, close = f, f.close

    try:
        emit_dbg_mtx(sigma, args.k, fh)
    finally:
        close()

if __name__ == "__main__":
    main()
