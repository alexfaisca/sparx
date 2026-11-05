// HyperBall iteration kernels.
//
// Counters are stored as in `HyperLogLog` (one byte per register), read as `u32` words holding 4
// registers each. As a single buffer may be too small for all counters, they are split in
// segments of `segment_nodes` counters: node `u`'s counter is `words` consecutive words starting
// at word `(u % segment_nodes) * words` of segment `u / segment_nodes`. The segments' bindings and
// accessors are generated (see `{{...}}` placeholders) for the number of segments needed.

struct Params {
    // number of nodes
    nodes: u32,
    // `u32` words per counter (registers / 4)
    words: u32,
    // largest register value, `64 - precision + 1`
    max_rank: u32,
    // counters per segment
    segment_nodes: u32,
    // nodes per row of `union_balls` invocations
    row_nodes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // current iteration's distance (`t`) and its inverse
    t: f32,
    inv_t: f32,
    // number of registers
    registers: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> offsets: array<u32>;
@group(0) @binding(1) var<storage, read> neighbours: array<u32>;
// per node: whether this iteration changed its counter
@group(0) @binding(2) var<storage, read_write> changed: array<atomic<u32>>;
// per node: (distance accumulator, inverse distance accumulator)
@group(0) @binding(3) var<storage, read_write> accumulators: array<vec2<f32>>;
// number of nodes whose counter changed this iteration
@group(0) @binding(4) var<storage, read_write> changed_count: atomic<u32>;
@group(0) @binding(5) var<uniform> params: Params;
// `current_k`: counters after the previous iteration, `next_k`: counters after this one
{{SEGMENT_BINDINGS}}

const WORKGROUP_SIZE: u32 = 64u;

// word `word` of node `node`'s counter within its segment
fn segment_word(node: u32, word: u32) -> u32 {
    return (node % params.segment_nodes) * params.words + word;
}

fn read_current(node: u32, word: u32) -> u32 {
    let i = segment_word(node, word);
    switch (node / params.segment_nodes) {
{{READ_CURRENT_CASES}}
        default: { return 0u; }
    }
}

fn read_next(node: u32, word: u32) -> u32 {
    let i = segment_word(node, word);
    switch (node / params.segment_nodes) {
{{READ_NEXT_CASES}}
        default: { return 0u; }
    }
}

fn write_next(node: u32, word: u32, value: u32) {
    let i = segment_word(node, word);
    switch (node / params.segment_nodes) {
{{WRITE_NEXT_CASES}}
        default: {}
    }
}

// byte-wise max of 4 registers
fn max_registers(a: u32, b: u32) -> u32 {
    var r = 0u;
    for (var s = 0u; s < 32u; s += 8u) {
        r |= max((a >> s) & 0xffu, (b >> s) & 0xffu) << s;
    }
    return r;
}

// one invocation per counter word, in rows of `row_nodes` whole counters (so that node indices
// never overflow): neighbours' words are read contiguously by consecutive invocations
@compute @workgroup_size(WORKGROUP_SIZE)
fn union_balls(@builtin(global_invocation_id) id: vec3<u32>) {
    let node = id.y * params.row_nodes + id.x / params.words;
    if (id.x / params.words >= params.row_nodes || node >= params.nodes) {
        return;
    }
    let word = id.x % params.words;
    let own = read_current(node, word);
    var ball = own;
    for (var e = offsets[node]; e < offsets[node + 1u]; e++) {
        ball = max_registers(ball, read_current(neighbours[e], word));
    }
    write_next(node, word, ball);
    if (ball != own) {
        atomicStore(&changed[node], 1u);
    }
}

fn sigma(x_in: f32) -> f32 {
    var x = x_in;
    var y = 1.0;
    var z = x;
    for (var k = 0u; k < 64u; k++) {
        x *= x;
        let prev = z;
        z += x * y;
        y += y;
        if (z == prev) {
            break;
        }
    }
    return z;
}

fn tau(x_in: f32) -> f32 {
    if (x_in == 0.0 || x_in == 1.0) {
        return 0.0;
    }
    var x = x_in;
    var y = 1.0;
    var z = 1.0 - x;
    for (var k = 0u; k < 64u; k++) {
        x = sqrt(x);
        let prev = z;
        y *= 0.5;
        z -= (1.0 - x) * (1.0 - x) * y;
        if (z == prev) {
            break;
        }
    }
    return z / 3.0;
}

// Same estimator as `HyperLogLog::estimate_cardinality` (Ertl's improved estimator with the
// finite register count correction), in `f32`, of `node`'s counter in `next` (`in_next`) or in
// `current`.
fn estimate(node: u32, in_next: bool) -> f32 {
    var sum = 0.0;
    var zeros = 0u;
    var saturated = 0u;
    for (var w = 0u; w < params.words; w++) {
        var word: u32;
        if (in_next) {
            word = read_next(node, w);
        } else {
            word = read_current(node, w);
        }
        for (var s = 0u; s < 32u; s += 8u) {
            let r = (word >> s) & 0xffu;
            sum += bitcast<f32>((127u - r) << 23u);
            zeros += select(0u, 1u, r == 0u);
            saturated += select(0u, 1u, r == params.max_rank);
        }
    }

    let m = params.registers;
    let z0 = f32(zeros);
    let zq = f32(saturated);
    if (z0 == m) {
        return 0.0;
    }
    let max_weight = bitcast<f32>((127u - params.max_rank) << 23u);
    let middle = sum - z0 - zq * max_weight;
    let z = m * tau(1.0 - zq / m) * (2.0 * max_weight) + middle + m * sigma(z0 / m);
    return m * m / (2.0 * 0.6931471805599453 * (1.0 + 1.079 / m) * z);
}

// one invocation per node (in rows of `num_workgroups.x * WORKGROUP_SIZE` nodes): accumulates the
// distances of the nodes its ball gained
@compute @workgroup_size(WORKGROUP_SIZE)
fn accumulate(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let node = id.x + id.y * groups.x * WORKGROUP_SIZE;
    if (node >= params.nodes || atomicLoad(&changed[node]) == 0u) {
        return;
    }
    atomicStore(&changed[node], 0u);
    atomicAdd(&changed_count, 1u);
    let diff = estimate(node, true) - estimate(node, false);
    if (diff > 0.0) {
        accumulators[node] += vec2<f32>(params.t * diff, params.inv_t * diff);
    }
}
