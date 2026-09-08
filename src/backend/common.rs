//! Shared GPU-kernel parameter layout, constants, and chunked dispatch helpers
//! for all GPU backends (Vulkan, OpenCL, HIP). The `Params` struct is uploaded
//! verbatim to the device as a uniform/constant buffer, so its field order and
//! size must stay in sync with the WGSL / OpenCL C / HIP kernels.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::sha::{write_decimal, Midstate};
use crate::task::{hex_lower, Solution, Task};

/// GPU workgroup (local) size used by every GPU backend. The WGSL shader's
/// `@workgroup_size(256)` and the OpenCL/HIP local size must match this.
pub const WORKGROUP: u32 = 256;

/// Maximum workgroups per single dispatch (Vulkan's 2^16-1 limit; OpenCL/HIP
/// use the same chunking for symmetry and to bound per-dispatch resources).
pub const MAX_WORKGROUPS: u32 = 65535;

/// Number of constant tail-block words uploaded to the device. A task's tail
/// message spans at most two 64-byte blocks (16 words each), so 32 words always
/// suffice. The digit region is left zeroed on the host and filled per-thread.
pub const CONSTANT_WORDS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "vulkan", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Params {
    pub nonce_start_lo: u32,
    pub nonce_start_hi: u32,
    pub count: u32,
    pub difficulty: u32,
    pub buffered: u32,
    pub nd: u32,
    pub num_blocks: u32,
    pub s0: u32,
    pub s1: u32,
    pub s2: u32,
    pub s3: u32,
    pub s4: u32,
    pub s5: u32,
    pub s6: u32,
    pub s7: u32,
}

impl Params {
    pub const SIZE: usize = std::mem::size_of::<Params>();

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chunk_start: u64,
        count: u32,
        difficulty: u32,
        buffered: usize,
        nd: u32,
        num_blocks: u32,
        state: [u32; 8],
    ) -> Self {
        Self {
            nonce_start_lo: chunk_start as u32,
            nonce_start_hi: (chunk_start >> 32) as u32,
            count,
            difficulty,
            buffered: buffered as u32,
            nd,
            num_blocks,
            s0: state[0],
            s1: state[1],
            s2: state[2],
            s3: state[3],
            s4: state[4],
            s5: state[5],
            s6: state[6],
            s7: state[7],
        }
    }
}

/// Number of decimal digits in `n`.
fn decimal_digits(n: u64) -> u32 {
    if n == 0 {
        return 1;
    }
    let mut d = 0u32;
    let mut x = n;
    while x > 0 {
        d += 1;
        x /= 10;
    }
    d
}

/// Smallest integer with `nd + 1` decimal digits (`10^nd`), i.e. the first
/// nonce whose digit count exceeds `nd`. Returns `u64::MAX` if it would
/// overflow (only relevant for 20-digit nonces).
fn pow10(exp: u32) -> u64 {
    let mut v = 1u64;
    for _ in 0..exp {
        v *= 10;
    }
    v
}

/// Build the constant tail-block words for a dispatch whose nonces all share
/// the same decimal digit count `nd`. The message is
/// `[partial bytes][nonce digits][0x80][zero padding][64-bit length]`; the
/// digit region is left zeroed and filled by the kernel per thread. Returns the
/// big-endian words for up to two 64-byte blocks plus the number of blocks.
pub fn build_tail_words(
    partial_block: &[u8; 64],
    buffered: usize,
    prefix_len: u64,
    nd: u32,
) -> ([u32; CONSTANT_WORDS], u32) {
    let nd = nd as usize;
    let total_len = prefix_len + nd as u64;
    let bitlen = total_len * 8;
    let mlen = buffered + nd + 1;
    let padded = (mlen + 8).div_ceil(64) * 64;
    let num_blocks = (padded / 64) as u32;

    let mut stream = [0u8; 128];
    stream[..buffered].copy_from_slice(&partial_block[..buffered]);
    stream[buffered + nd] = 0x80;
    stream[padded - 8..padded].copy_from_slice(&bitlen.to_be_bytes());

    let mut words = [0u32; CONSTANT_WORDS];
    for (wi, chunk) in stream.chunks(4).enumerate() {
        let mut w = 0u32;
        for &b in chunk {
            w = (w << 8) | b as u32;
        }
        words[wi] = w;
    }
    (words, num_blocks)
}

/// Recompute the SHA-256 hash for a found nonce and build the `Solution` to
/// submit. The GPU kernels only read back the nonce (not the hash), so this is
/// verified on the host against the original prefix.
pub fn make_solution(task: &Task, prefix_bytes: &[u8], nonce: u64) -> Solution {
    let mut mid = Midstate::new();
    mid.update(prefix_bytes);
    let mut digits = [0u8; 20];
    let n = write_decimal(nonce, &mut digits);
    let hash = mid.finish(&digits[..n]);
    Solution {
        task_id: task.id.clone(),
        nonce,
        hash: hex_lower(&hash),
    }
}

/// Outcome of a single chunk dispatch, reported back by the backend-specific
/// closure passed to [`mine_chunks`].
pub enum ChunkOutcome {
    NotFound,
    Found(u64),
    Abort,
}

/// Drive the shared chunked-dispatch loop over a nonce range, invoking `f` once
/// per chunk. Each chunk is capped to a single dispatch and is additionally
/// clamped so every nonce in it shares the same decimal digit count `nd` (the
/// digit count is passed to `f`). `f` uploads params + constant words, launches
/// the kernel, and reads back the result. Returns the winning nonce if found,
/// or `None` on exhaustion/abort.
pub fn mine_chunks(
    start: u64,
    count: u64,
    counter: &AtomicU64,
    mut f: impl FnMut(u64, u32, u32) -> ChunkOutcome,
) -> Option<u64> {
    let threads_per_dispatch = (WORKGROUP * MAX_WORKGROUPS) as u64;
    let mut off = 0u64;
    let mut remaining = count;
    while remaining > 0 {
        let chunk_start = start + off;
        let nd = decimal_digits(chunk_start);
        // Clamp so the chunk never crosses a power-of-ten boundary (otherwise
        // `nd` would not be uniform within the dispatch).
        let digit_limit = if nd >= 20 {
            u64::MAX
        } else {
            pow10(nd) - chunk_start
        };
        let c = remaining.min(threads_per_dispatch).min(digit_limit) as u32;
        counter.fetch_add(c as u64, Ordering::Relaxed);
        match f(chunk_start, c, nd) {
            ChunkOutcome::Found(nonce) => return Some(nonce),
            ChunkOutcome::Abort => return None,
            ChunkOutcome::NotFound => {}
        }
        off += c as u64;
        remaining -= c as u64;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha::{write_decimal, Midstate};

    /// Reproduce the GPU kernel's tail-message construction + compression in
    /// Rust and return the resulting compression state.
    fn kernel_hash(prefix: &[u8], nonce: u64) -> [u32; 8] {
        let mut mid = Midstate::new();
        mid.update(prefix);
        let (state, partial_block, buffered) = mid.midstate_parts();
        let prefix_len = mid.total_len();
        let nd = decimal_digits(nonce);
        let (mut words, num_blocks) = build_tail_words(&partial_block, buffered, prefix_len, nd);

        // Overlay the nonce's decimal digits, least-significant first.
        let mut cur = nonce;
        for i in 0..nd {
            let digit = (cur % 10) as u32;
            cur /= 10;
            let pos = buffered as u32 + (nd - 1 - i);
            let wi = (pos >> 2) as usize;
            let shift = 24 - (pos & 3) * 8;
            words[wi] |= (0x30 + digit) << shift;
        }

        let mut h = state;
        for b in 0..num_blocks as usize {
            let mut block = [0u8; 64];
            for t in 0..16 {
                let w = words[b * 16 + t];
                block[4 * t..4 * t + 4].copy_from_slice(&w.to_be_bytes());
            }
            sha2::block_api::compress256(&mut h, &[block]);
        }
        h
    }

    #[test]
    fn tail_words_match_reference() {
        let prefix = b"task_60346_diff_2_{whatever}_";
        let nonces = [
            0u64,
            1,
            9,
            10,
            99,
            12345,
            999_999_999,
            1_000_000_000,
            5_000_000_000,
            5_010_000_000,
            10_000_000_000_000_000_000,
            u64::MAX,
        ];
        for nonce in nonces {
            let mut mid = Midstate::new();
            mid.update(prefix);
            let mut digits = [0u8; 20];
            let n = write_decimal(nonce, &mut digits);
            let expected = mid.finish_raw(&digits[..n]);
            assert_eq!(kernel_hash(prefix, nonce), expected, "nonce {nonce}");
        }
    }

    #[test]
    fn chunks_keep_uniform_digit_count() {
        // A range crossing several power-of-ten boundaries must be split so
        // every chunk has a uniform digit count and full coverage.
        let start = 998_000u64;
        let count = 5_000u64; // crosses 10^6, 10^7 boundaries? No: 998000..1003000 crosses 10^6.
        let counter = AtomicU64::new(0);
        let mut seen = Vec::new();
        mine_chunks(start, count, &counter, |chunk_start, c, nd| {
            assert_eq!(decimal_digits(chunk_start), nd);
            let end = chunk_start + c as u64 - 1;
            assert_eq!(decimal_digits(end), nd, "chunk crosses a digit boundary");
            seen.push((chunk_start, c, nd));
            ChunkOutcome::NotFound
        });
        assert!(!seen.is_empty());
        let total: u64 = seen.iter().map(|&(_, c, _)| c as u64).sum();
        assert_eq!(total, count);
        // The chunk covering 999_999 must have nd == 6, the one covering
        // 1_000_000 must have nd == 7.
        let max_nd = seen.iter().map(|&(_, _, nd)| nd).max().unwrap();
        let min_nd = seen.iter().map(|&(_, _, nd)| nd).min().unwrap();
        assert!(max_nd > min_nd, "range should have crossed a digit boundary");
    }
}
