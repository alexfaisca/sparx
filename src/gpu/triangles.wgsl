// Per-edge triangle counting.
//
// Neighbour lists are sorted, so every triangle `x < y < z` is found exactly once, by the
// invocation of its lowest edge `x -> y`, by merging `x`'s and `y`'s neighbours above `y`. It then
// counts once for each of its 6 (directed) edges --- once per combination of parallel edges, as
// `Triangles` does on the CPU.

struct Params {
    nodes: u32,
    edges: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> offsets: array<u32>;
@group(0) @binding(1) var<storage, read> neighbours: array<u32>;
// per node: offset of its first neighbour above it
@group(0) @binding(2) var<storage, read> edge_over: array<u32>;
// per edge `u -> v`: the offset of `v -> u`
@group(0) @binding(3) var<storage, read> reciprocal: array<u32>;
// per edge: triangles it is in
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 64u;

// the node whose neighbour list holds edge `e`: the last `x` with `offsets[x] <= e`
fn source(e: u32) -> u32 {
    var lo = 0u;
    var hi = params.nodes - 1u;
    while (lo < hi) {
        let mid = lo + (hi - lo + 1u) / 2u;
        if (offsets[mid] <= e) {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    return lo;
}

fn count(e: u32) {
    atomicAdd(&counts[e], 1u);
    atomicAdd(&counts[reciprocal[e]], 1u);
}

// one invocation per edge (in rows of `num_workgroups.x * WORKGROUP_SIZE` edges)
@compute @workgroup_size(WORKGROUP_SIZE)
fn count_triangles(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let e = id.x + id.y * groups.x * WORKGROUP_SIZE;
    if (e >= params.edges) {
        return;
    }
    let x = source(e);
    let y = neighbours[e];
    if (y <= x) {
        return;
    }

    // `x`'s neighbours above `y` (past `y` itself and edges parallel to `e`) and `y`'s above `y`
    var p = e + 1u;
    let p_end = offsets[x + 1u];
    while (p < p_end && neighbours[p] == y) {
        p++;
    }
    var q = edge_over[y];
    let q_end = offsets[y + 1u];

    while (p < p_end && q < q_end) {
        let a = neighbours[p];
        let b = neighbours[q];
        if (a < b) {
            p++;
        } else if (a > b) {
            q++;
        } else {
            // runs of parallel edges to the common neighbour `a`
            var p_run = p;
            while (p_run < p_end && neighbours[p_run] == a) {
                p_run++;
            }
            var q_run = q;
            while (q_run < q_end && neighbours[q_run] == a) {
                q_run++;
            }
            for (var i = p; i < p_run; i++) {
                for (var j = q; j < q_run; j++) {
                    count(e);
                    count(i);
                    count(j);
                }
            }
            p = p_run;
            q = q_run;
        }
    }
}
