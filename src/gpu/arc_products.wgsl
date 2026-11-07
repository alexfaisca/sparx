// The sum, over every arc `u -> v`, of `x[u] * y[v]`: assortativity's only sum over arcs.
//
// One invocation per node `u` sums its heads' `y` (in 64 bits) and multiplies the sum by `x[u]`
// (in 96 bits); each workgroup then adds its invocations' products up (in 128 bits) and writes
// its sum, which the host adds up. WGSL has no 64 bit integers: every wide value is little
// endian 32 bit limbs, and nothing overflows (arcs are fewer than 2^32, values below 2^32).

struct Params {
    nodes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> offsets: array<u32>;
@group(0) @binding(1) var<storage, read> neighbours: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<u32>;
@group(0) @binding(3) var<storage, read> y: array<u32>;
// per workgroup: its sum, as 4 limbs
@group(0) @binding(4) var<storage, read_write> partials: array<vec4<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> scratch: array<vec4<u32>, 64>;

// `a * b`, as (low, high) limbs
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let lo = a0 * b0;
    let mid1 = a1 * b0;
    let mid2 = a0 * b1;
    let hi = a1 * b1;
    let mid = mid1 + mid2;
    // carry of mid1 + mid2, worth 2^48 overall: 2^16 in the high limb
    let mid_carry = select(0u, 0x10000u, mid < mid1);
    let low = lo + (mid << 16u);
    let low_carry = select(0u, 1u, low < lo);
    return vec2<u32>(low, hi + (mid >> 16u) + mid_carry + low_carry);
}

// `a + b`, 128 bit
fn add128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var out: vec4<u32>;
    var carry = 0u;
    for (var i = 0u; i < 4u; i++) {
        let s = a[i] + b[i];
        let c1 = select(0u, 1u, s < a[i]);
        let t = s + carry;
        let c2 = select(0u, 1u, t < s);
        out[i] = t;
        carry = c1 + c2;
    }
    return out;
}

@compute @workgroup_size(64)
fn arc_products(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let u = id.x + id.y * groups.x * WORKGROUP_SIZE;
    var product = vec4<u32>(0u, 0u, 0u, 0u);
    if (u < params.nodes) {
        // the heads' values' sum, 64 bit
        var lo = 0u;
        var hi = 0u;
        let end = offsets[u + 1u];
        for (var e = offsets[u]; e < end; e++) {
            let v = y[neighbours[e]];
            let s = lo + v;
            hi += select(0u, 1u, s < lo);
            lo = s;
        }
        // times x[u]: x * lo + (x * hi) << 32
        let xu = x[u];
        let p0 = mul32(xu, lo);
        let p1 = mul32(xu, hi);
        product = add128(vec4<u32>(p0.x, p0.y, 0u, 0u), vec4<u32>(0u, p1.x, p1.y, 0u));
    }
    scratch[local] = product;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (local < stride) {
            scratch[local] = add128(scratch[local], scratch[local + stride]);
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        partials[group.x + group.y * groups.x] = scratch[0];
    }
}
