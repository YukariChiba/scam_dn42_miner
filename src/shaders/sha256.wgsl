// SHA-256 PoW miner compute shader (midstate variant, u64 nonce).
//
// The fixed per-task prefix is pre-compressed on the host into an 8-word state
// plus the constant tail-block words (`constant_w`) with the nonce-digit region
// zeroed. Each invocation computes the decimal digits of its nonce, overlays
// them into the constant words, and compresses only the remaining tail
// block(s). Nonces are 64-bit (two u32 halves), so difficulty >= 7 ranges work.
//
// The tail message construction is done at word granularity: the host bakes the
// partial bytes, 0x80 pad, zero padding, and the 64-bit length field into
// `constant_w`, so the kernel only does `nd` (<= 20) digit-overlay operations
// instead of 64 per-byte `tail_byte` calls.

struct Params {
    nonce_start_lo: u32,
    nonce_start_hi: u32,
    count: u32,
    difficulty: u32,
    buffered: u32,
    nd: u32,
    num_blocks: u32,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    s4: u32,
    s5: u32,
    s6: u32,
    s7: u32,
};

struct DivMod10 {
    q_lo: u32,
    q_hi: u32,
    rem: u32,
};

struct U20DivMod {
    q: u32,
    rem: u32,
};

@group(0) @binding(0) var<storage, read> constant_w: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

// Compile-time constant table: stays in constant/uniform memory instead of
// consuming per-thread private registers (256 bytes per invocation).
const K = array<u32, 64>(
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
);

fn rotr(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

// Divide a value < 2^20 by 10 using reciprocal multiplication (no / or %).
// `cur = hi * 1024 + lo`, and 1024 = 10 * 102 + 4, so:
//   floor(cur/10) = hi * 102 + floor((hi*4 + lo) / 10)
//   cur % 10      = (hi*4 + lo) % 10
// The inner divide is on a value < 5120, done as (v * 13108) >> 17.
fn divmod10_u20(cur: u32) -> U20DivMod {
    let hi = cur >> 10u;
    let lo = cur & 0x3FFu;
    let tmp = hi * 4u + lo;
    let q_rem = (tmp * 13108u) >> 17u;
    let rem = tmp - q_rem * 10u;
    return U20DivMod(hi * 102u + q_rem, rem);
}

fn divmod10(lo: u32, hi: u32) -> DivMod10 {
    let a3 = hi >> 16u;
    let a2 = hi & 0xFFFFu;
    let a1 = lo >> 16u;
    let a0 = lo & 0xFFFFu;

    var rem = 0u;
    var cur = a3;
    var d = divmod10_u20(cur);
    let q3 = d.q;
    rem = d.rem;

    cur = (rem << 16u) | a2;
    d = divmod10_u20(cur);
    let q2 = d.q;
    rem = d.rem;

    cur = (rem << 16u) | a1;
    d = divmod10_u20(cur);
    let q1 = d.q;
    rem = d.rem;

    cur = (rem << 16u) | a0;
    d = divmod10_u20(cur);
    let q0 = d.q;
    rem = d.rem;

    return DivMod10((q1 << 16u) | q0, (q3 << 16u) | q2, rem);
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    // nonce = nonce_start (u64) + idx (u32)
    let nlo = params.nonce_start_lo + idx;
    var nhi = params.nonce_start_hi;
    if (nlo < params.nonce_start_lo) {
        nhi = nhi + 1u;
    }

    // Copy the constant tail-block words, then overlay this thread's decimal
    // digits (least-significant first) into the zeroed digit region.
    var w: array<u32, 32>;
    for (var t = 0u; t < 16u; t = t + 1u) {
        w[t] = constant_w[t];
    }
    if (params.num_blocks == 2u) {
        for (var t = 0u; t < 16u; t = t + 1u) {
            w[16u + t] = constant_w[16u + t];
        }
    }

    var cur_lo = nlo;
    var cur_hi = nhi;
    for (var i = 0u; i < params.nd; i = i + 1u) {
        let r = divmod10(cur_lo, cur_hi);
        let digit = r.rem;
        cur_lo = r.q_lo;
        cur_hi = r.q_hi;
        // Position of this digit in the tail stream (least significant last).
        let pos = params.buffered + (params.nd - 1u - i);
        let wi = pos >> 2u;
        let shift = 24u - (pos & 3u) * 8u;
        w[wi] = w[wi] | ((0x30u + digit) << shift);
    }

    // Compress the tail block(s) starting from the pre-computed midstate.
    var h0 = params.s0;
    var h1 = params.s1;
    var h2 = params.s2;
    var h3 = params.s3;
    var h4 = params.s4;
    var h5 = params.s5;
    var h6 = params.s6;
    var h7 = params.s7;

    for (var b = 0u; b < params.num_blocks; b = b + 1u) {
        // Rolling 16-word message-schedule window, seeded from the block words.
        var m: array<u32, 16>;
        for (var t = 0u; t < 16u; t = t + 1u) {
            m[t] = w[b * 16u + t];
        }

        var wa = h0;
        var wb = h1;
        var wc = h2;
        var wd = h3;
        var we = h4;
        var wf = h5;
        var wg = h6;
        var wh = h7;

        for (var t = 0u; t < 64u; t = t + 1u) {
            if (t >= 16u) {
                let sig0 = rotr(m[(t - 15u) & 15u], 7u) ^ rotr(m[(t - 15u) & 15u], 18u) ^ (m[(t - 15u) & 15u] >> 3u);
                let sig1 = rotr(m[(t - 2u) & 15u], 17u) ^ rotr(m[(t - 2u) & 15u], 19u) ^ (m[(t - 2u) & 15u] >> 10u);
                m[t & 15u] = m[t & 15u] + sig0 + m[(t - 7u) & 15u] + sig1;
            }
            let s1 = rotr(we, 6u) ^ rotr(we, 11u) ^ rotr(we, 25u);
            let ch = (we & wf) ^ ((~we) & wg);
            let temp1 = wh + s1 + ch + K[t] + m[t & 15u];
            let s0 = rotr(wa, 2u) ^ rotr(wa, 13u) ^ rotr(wa, 22u);
            let maj = (wa & wb) ^ (wa & wc) ^ (wb & wc);

            wh = wg;
            wg = wf;
            wf = we;
            we = wd + temp1;
            wd = wc;
            wc = wb;
            wb = wa;
            wa = temp1 + s0 + maj;
        }

        h0 = h0 + wa;
        h1 = h1 + wb;
        h2 = h2 + wc;
        h3 = h3 + wd;
        h4 = h4 + we;
        h5 = h5 + wf;
        h6 = h6 + wg;
        h7 = h7 + wh;
    }

    // Leading-zero check.
    var zbits = 0u;
    var matched = false;
    var hv: array<u32, 8>;
    hv[0] = h0;
    hv[1] = h1;
    hv[2] = h2;
    hv[3] = h3;
    hv[4] = h4;
    hv[5] = h5;
    hv[6] = h6;
    hv[7] = h7;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let clz = countLeadingZeros(hv[i]);
        zbits = zbits + clz;
        if (clz < 32u) {
            break;
        }
    }
    if (zbits >= params.difficulty * 4u) {
        matched = true;
    }

    if (matched) {
        let res = atomicCompareExchangeWeak(&result[2u], 0u, 1u);
        if (res.exchanged) {
            atomicStore(&result[0u], nlo);
            atomicStore(&result[1u], nhi);
        }
    }
}
