// HyperBall iteration kernels (CUDA): `hyperball.wgsl`'s, compiled at runtime by NVRTC.
//
// Counters are stored as in `HyperLogLog` (one byte per register), read as 32-bit words holding 4
// registers each: node `u`'s counter is the `words` words from word `u * words` on. CUDA buffers
// being as large as the device's memory, counters are in one buffer (`hyperball.wgsl` splits them
// in segments, wgpu's buffers being smaller).

// byte-wise max of 4 registers
__device__ __forceinline__ unsigned int max_registers(unsigned int a, unsigned int b) {
    unsigned int r = 0u;
    for (unsigned int s = 0u; s < 32u; s += 8u) {
        r |= max((a >> s) & 0xffu, (b >> s) & 0xffu) << s;
    }
    return r;
}

// one thread per counter word: neighbours' words are read contiguously by consecutive threads
extern "C" __global__ void union_balls(const unsigned int* __restrict__ offsets,
                                       const unsigned int* __restrict__ neighbours,
                                       const unsigned int* __restrict__ current,
                                       unsigned int* __restrict__ next,
                                       unsigned int* __restrict__ changed, unsigned int nodes,
                                       unsigned int words) {
    unsigned long long total = (unsigned long long)nodes * words;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < total; i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int node = (unsigned int)(i / words);
        unsigned int word = (unsigned int)(i % words);
        unsigned int own = current[i];
        unsigned int ball = own;
        for (unsigned int e = offsets[node]; e < offsets[node + 1u]; e++) {
            ball = max_registers(ball, current[(unsigned long long)neighbours[e] * words + word]);
        }
        next[i] = ball;
        if (ball != own) {
            changed[node] = 1u;
        }
    }
}

__device__ float sigma(float x) {
    float y = 1.0f;
    float z = x;
    for (unsigned int k = 0u; k < 64u; k++) {
        x *= x;
        float prev = z;
        z += x * y;
        y += y;
        if (z == prev) {
            break;
        }
    }
    return z;
}

__device__ float tau(float x) {
    if (x == 0.0f || x == 1.0f) {
        return 0.0f;
    }
    float y = 1.0f;
    float z = 1.0f - x;
    for (unsigned int k = 0u; k < 64u; k++) {
        x = sqrtf(x);
        float prev = z;
        y *= 0.5f;
        z -= (1.0f - x) * (1.0f - x) * y;
        if (z == prev) {
            break;
        }
    }
    return z / 3.0f;
}

// Same estimator as `HyperLogLog::estimate_cardinality` (Ertl's improved estimator with the
// finite register count correction), in `f32`, of the counter at `counter`.
__device__ float estimate(const unsigned int* counter, unsigned int words, unsigned int max_rank,
                          float registers) {
    float sum = 0.0f;
    unsigned int zeros = 0u;
    unsigned int saturated = 0u;
    for (unsigned int w = 0u; w < words; w++) {
        unsigned int word = counter[w];
        for (unsigned int s = 0u; s < 32u; s += 8u) {
            unsigned int r = (word >> s) & 0xffu;
            sum += __uint_as_float((127u - r) << 23u);
            zeros += r == 0u ? 1u : 0u;
            saturated += r == max_rank ? 1u : 0u;
        }
    }

    float m = registers;
    float z0 = (float)zeros;
    float zq = (float)saturated;
    if (z0 == m) {
        return 0.0f;
    }
    float max_weight = __uint_as_float((127u - max_rank) << 23u);
    float middle = sum - z0 - zq * max_weight;
    float z = m * tau(1.0f - zq / m) * (2.0f * max_weight) + middle + m * sigma(z0 / m);
    return m * m / (2.0f * 0.6931471805599453f * (1.0f + 1.079f / m) * z);
}

// one thread per node: accumulates the distances of the nodes its ball gained
extern "C" __global__ void accumulate(const unsigned int* __restrict__ current,
                                      const unsigned int* __restrict__ next,
                                      unsigned int* __restrict__ changed,
                                      float2* __restrict__ accumulators,
                                      unsigned int* __restrict__ changed_count, unsigned int nodes,
                                      unsigned int words, unsigned int max_rank, float registers,
                                      float t, float inv_t) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < nodes; i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int node = (unsigned int)i;
        if (changed[node] == 0u) {
            continue;
        }
        changed[node] = 0u;
        atomicAdd(changed_count, 1u);
        unsigned long long first = (unsigned long long)node * words;
        float diff = estimate(next + first, words, max_rank, registers) -
                     estimate(current + first, words, max_rank, registers);
        if (diff > 0.0f) {
            accumulators[node].x += t * diff;
            accumulators[node].y += inv_t * diff;
        }
    }
}
