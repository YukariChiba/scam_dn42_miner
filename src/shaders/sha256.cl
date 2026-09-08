// SHA-256 PoW miner compute kernel (midstate variant, u64 nonce).
// OpenCL C port of sha256.wgsl. The fixed per-task prefix is pre-compressed on
// the host into an 8-word state plus the constant tail-block words
// (`constant_w`) with the nonce-digit region zeroed. Each invocation computes
// the decimal digits of its nonce, overlays them into the constant words, and
// compresses only the remaining tail block(s). Nonces are 64-bit (two u32
// halves).

struct Params {
    uint nonce_start_lo;
    uint nonce_start_hi;
    uint count;
    uint difficulty;
    uint buffered;
    uint nd;
    uint num_blocks;
    uint s0;
    uint s1;
    uint s2;
    uint s3;
    uint s4;
    uint s5;
    uint s6;
    uint s7;
};

struct DivMod10 {
    uint q_lo;
    uint q_hi;
    uint rem;
};

struct U20DivMod {
    uint q;
    uint rem;
};

// Compile-time constant table: kept in constant memory instead of consuming
// per-thread private registers.
__constant uint K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

// Divide a value < 2^20 by 10 using reciprocal multiplication (no / or %).
struct U20DivMod divmod10_u20(uint cur) {
    uint hi = cur >> 10u;
    uint lo = cur & 0x3FFu;
    uint tmp = hi * 4u + lo;
    uint q_rem = (tmp * 13108u) >> 17u;
    uint rem = tmp - q_rem * 10u;
    struct U20DivMod r;
    r.q = hi * 102u + q_rem;
    r.rem = rem;
    return r;
}

struct DivMod10 divmod10(uint lo, uint hi) {
    uint a3 = hi >> 16u;
    uint a2 = hi & 0xFFFFu;
    uint a1 = lo >> 16u;
    uint a0 = lo & 0xFFFFu;

    uint rem = 0u;
    uint cur = a3;
    struct U20DivMod d = divmod10_u20(cur);
    uint q3 = d.q;
    rem = d.rem;

    cur = (rem << 16u) | a2;
    d = divmod10_u20(cur);
    uint q2 = d.q;
    rem = d.rem;

    cur = (rem << 16u) | a1;
    d = divmod10_u20(cur);
    uint q1 = d.q;
    rem = d.rem;

    cur = (rem << 16u) | a0;
    d = divmod10_u20(cur);
    uint q0 = d.q;
    rem = d.rem;

    struct DivMod10 r;
    r.q_lo = (q1 << 16u) | q0;
    r.q_hi = (q3 << 16u) | q2;
    r.rem = rem;
    return r;
}

__kernel void mine(__global const uint* constant_w,
                   __global volatile uint* result,
                   __global const struct Params* params) {
    uint idx = get_global_id(0);
    if (idx >= params->count) {
        return;
    }

    // nonce = nonce_start (u64) + idx (u32)
    uint nlo = params->nonce_start_lo + idx;
    uint nhi = params->nonce_start_hi;
    if (nlo < params->nonce_start_lo) {
        nhi = nhi + 1u;
    }

    // Copy the constant tail-block words, then overlay this thread's decimal
    // digits (least-significant first) into the zeroed digit region.
    uint w[32];
    #pragma unroll
    for (uint t = 0u; t < 16u; t = t + 1u) {
        w[t] = constant_w[t];
    }
    if (params->num_blocks == 2u) {
        #pragma unroll
        for (uint t = 0u; t < 16u; t = t + 1u) {
            w[16u + t] = constant_w[16u + t];
        }
    }

    uint cur_lo = nlo;
    uint cur_hi = nhi;
    for (uint i = 0u; i < params->nd; i = i + 1u) {
        struct DivMod10 r = divmod10(cur_lo, cur_hi);
        uint digit = r.rem;
        cur_lo = r.q_lo;
        cur_hi = r.q_hi;
        uint pos = params->buffered + (params->nd - 1u - i);
        uint wi = pos >> 2u;
        uint shift = 24u - (pos & 3u) * 8u;
        w[wi] = w[wi] | ((0x30u + digit) << shift);
    }

    // Compress the tail block(s) starting from the pre-computed midstate.
    uint h0 = params->s0;
    uint h1 = params->s1;
    uint h2 = params->s2;
    uint h3 = params->s3;
    uint h4 = params->s4;
    uint h5 = params->s5;
    uint h6 = params->s6;
    uint h7 = params->s7;

    for (uint b = 0u; b < params->num_blocks; b = b + 1u) {
        uint w64[64];
        #pragma unroll
        for (uint t = 0u; t < 16u; t = t + 1u) {
            w64[t] = w[b * 16u + t];
        }
        #pragma unroll
        for (uint t = 16u; t < 64u; t = t + 1u) {
            uint s0 = ((w64[t - 15u] >> 7u) | (w64[t - 15u] << 25u)) ^
                      ((w64[t - 15u] >> 18u) | (w64[t - 15u] << 14u)) ^
                      (w64[t - 15u] >> 3u);
            uint s1 = ((w64[t - 2u] >> 17u) | (w64[t - 2u] << 15u)) ^
                      ((w64[t - 2u] >> 19u) | (w64[t - 2u] << 13u)) ^
                      (w64[t - 2u] >> 10u);
            w64[t] = w64[t - 16u] + s0 + w64[t - 7u] + s1;
        }

        uint a = h0;
        uint b0 = h1;
        uint c = h2;
        uint d = h3;
        uint e = h4;
        uint f = h5;
        uint g = h6;
        uint h = h7;

        #pragma unroll
        for (uint t = 0u; t < 64u; t = t + 1u) {
            uint s1 = ((e >> 6u) | (e << 26u)) ^
                      ((e >> 11u) | (e << 21u)) ^
                      ((e >> 25u) | (e << 7u));
            uint ch = (e & f) ^ ((~e) & g);
            uint temp1 = h + s1 + ch + K[t] + w64[t];
            uint s0 = ((a >> 2u) | (a << 30u)) ^
                      ((a >> 13u) | (a << 19u)) ^
                      ((a >> 22u) | (a << 10u));
            uint maj = (a & b0) ^ (a & c) ^ (b0 & c);

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b0;
            b0 = a;
            a = temp1 + s0 + maj;
        }

        h0 = h0 + a;
        h1 = h1 + b0;
        h2 = h2 + c;
        h3 = h3 + d;
        h4 = h4 + e;
        h5 = h5 + f;
        h6 = h6 + g;
        h7 = h7 + h;
    }

    // Leading-zero check.
    uint zbits = 0u;
    bool matched = false;
    uint hv[8];
    hv[0] = h0;
    hv[1] = h1;
    hv[2] = h2;
    hv[3] = h3;
    hv[4] = h4;
    hv[5] = h5;
    hv[6] = h6;
    hv[7] = h7;
    for (uint i = 0u; i < 8u; i = i + 1u) {
        uint lz = clz(hv[i]);
        zbits = zbits + lz;
        if (lz < 32u) {
            break;
        }
    }
    if (zbits >= params->difficulty * 4u) {
        matched = true;
    }

    if (matched) {
        if (atomic_cmpxchg(&result[2], 0u, 1u) == 0u) {
            atomic_xchg(&result[0], nlo);
            atomic_xchg(&result[1], nhi);
        }
    }
}
