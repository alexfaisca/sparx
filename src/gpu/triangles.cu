// Per-edge triangle counting (CUDA): `triangles.wgsl`'s kernel, compiled at runtime by NVRTC.
//
// Neighbour lists are sorted, so every triangle `x < y < z` is found exactly once, by the thread of
// its lowest edge `x -> y`, by merging `x`'s and `y`'s neighbours above `y`. It then counts once for
// each of its 3 upward edges (`x -> y`, `x -> z` and `y -> z`) --- once per combination of parallel
// edges, as `Triangles` does on the CPU. Downward edges are left at 0: the host copies their
// reciprocals' counts.
//
// Counts are `COUNT_BITS` wide (8, 16 or 32, defined by the host), packed little-endian in 32-bit
// words: the words' bytes are then those of the host's array of counts, read back as is. A count
// already at its largest value when added to sets `overflow` (its carry may then have spoilt its
// neighbours', which the host then discards).

#ifndef COUNT_BITS
#error "COUNT_BITS must be defined (8, 16 or 32)"
#endif

#define PER_WORD (32u / COUNT_BITS)
#define COUNT_MAX (0xffffffffu >> (32u - COUNT_BITS))

// the node whose neighbour list holds edge `e`: the last `x` with `offsets[x] <= e`
__device__ __forceinline__ unsigned int source(const unsigned int* __restrict__ offsets,
                                              unsigned int nodes, unsigned int e) {
    unsigned int lo = 0u;
    unsigned int hi = nodes - 1u;
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo + 1u) / 2u;
        if (offsets[mid] <= e) {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    return lo;
}

__device__ __forceinline__ void count(unsigned int* counts, unsigned int* overflow, unsigned int e) {
    unsigned int shift = (e % PER_WORD) * COUNT_BITS;
    unsigned int old = atomicAdd(&counts[e / PER_WORD], 1u << shift);
    if (((old >> shift) & COUNT_MAX) == COUNT_MAX) {
        atomicExch(overflow, 1u);
    }
}

extern "C" __global__ void count_triangles(const unsigned int* __restrict__ offsets,
                                           const unsigned int* __restrict__ neighbours,
                                           const unsigned int* __restrict__ edge_over,
                                           unsigned int* counts, unsigned int* overflow,
                                           unsigned int nodes, unsigned int edges) {
    // a grid-stride loop: any grid covers every edge
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < edges; i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int e = (unsigned int)i;
        unsigned int x = source(offsets, nodes, e);
        unsigned int y = neighbours[e];
        if (y <= x) {
            continue;
        }

        // `x`'s neighbours above `y` (past `y` itself and edges parallel to `e`) and `y`'s above `y`
        unsigned int p = e + 1u;
        unsigned int p_end = offsets[x + 1u];
        while (p < p_end && neighbours[p] == y) {
            p++;
        }
        unsigned int q = edge_over[y];
        unsigned int q_end = offsets[y + 1u];

        while (p < p_end && q < q_end) {
            unsigned int a = neighbours[p];
            unsigned int b = neighbours[q];
            if (a < b) {
                p++;
            } else if (a > b) {
                q++;
            } else {
                // runs of parallel edges to the common neighbour `a`
                unsigned int p_run = p;
                while (p_run < p_end && neighbours[p_run] == a) {
                    p_run++;
                }
                unsigned int q_run = q;
                while (q_run < q_end && neighbours[q_run] == a) {
                    q_run++;
                }
                for (unsigned int i = p; i < p_run; i++) {
                    for (unsigned int j = q; j < q_run; j++) {
                        count(counts, overflow, e);
                        count(counts, overflow, i);
                        count(counts, overflow, j);
                    }
                }
                p = p_run;
                q = q_run;
            }
        }
    }
}
