//! SHA-256 midstate helper: compresses the fixed per-task prefix once and
//! reuses that state, so only a single tail block is compressed per nonce.
//! `sha2::compress256` is hardware accelerated (SHA-NI / ARMv8) when available.

use sha2::block_api::compress256 as compress;

const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
    0x5be0cd19,
];

pub struct Midstate {
    state: [u32; 8],
    buf: [u8; 64],
    buffered: usize,
    total: u64,
}

impl Midstate {
    pub fn new() -> Self {
        Self {
            state: IV,
            buf: [0; 64],
            buffered: 0,
            total: 0,
        }
    }

    /// Bytes already consumed by `update` (used for the final length field).
    pub fn total_len(&self) -> u64 {
        self.total
    }

    /// Expose the compression state and the partial block for GPU midstate.
    /// Returns `(state[8], partial_buffer[64], buffered_count)`.
    pub fn midstate_parts(&self) -> ([u32; 8], [u8; 64], usize) {
        (self.state, self.buf, self.buffered)
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        while !data.is_empty() {
            let n = (64 - self.buffered).min(data.len());
            self.buf[self.buffered..self.buffered + n].copy_from_slice(&data[..n]);
            self.buffered += n;
            data = &data[n..];
            if self.buffered == 64 {
                compress(&mut self.state, &[self.buf]);
                self.buffered = 0;
            }
        }
    }

    /// Finish the digest with `tail` appended, returning the raw compression
    /// state `[u32; 8]` instead of serializing to bytes. The leading bits of
    /// the digest are exactly the leading bits of `state[0]`, so difficulty
    /// checks can be done on the register values directly.
    ///
    /// Does not mutate `self`, so it can be shared immutably across threads.
    #[inline(always)]
    pub fn finish_raw(&self, tail: &[u8]) -> [u32; 8] {
        let mut state = self.state;
        let mut buf = self.buf;
        let mut pos = self.buffered;

        let mut t = tail;
        while !t.is_empty() {
            if pos == 64 {
                compress(&mut state, &[buf]);
                buf = [0u8; 64];
                pos = 0;
            }
            let n = (64 - pos).min(t.len());
            buf[pos..pos + n].copy_from_slice(&t[..n]);
            pos += n;
            t = &t[n..];
        }

        buf[pos] = 0x80;
        pos += 1;
        if pos > 56 {
            compress(&mut state, &[buf]);
            buf = [0u8; 64];
            pos = 0;
        }
        // Zero the padding gap, which may otherwise contain stale buffer bytes.
        buf[pos..56].fill(0);
        let bit_len = (self.total + tail.len() as u64) * 8;
        buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        compress(&mut state, &[buf]);

        state
    }

    /// Finish the digest with `tail` appended (e.g. the decimal nonce).
    pub fn finish(&self, tail: &[u8]) -> [u8; 32] {
        state_to_bytes(&self.finish_raw(tail))
    }
}

/// Serialize the big-endian compression state into a 32-byte digest.
#[inline(always)]
pub fn state_to_bytes(state: &[u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in state.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
    }
    out
}

/// Fast leading-zero check directly on the SHA-256 state words, without ever
/// serializing the digest to a byte array. The digest is the big-endian
/// concatenation of `state[0..8]`, so `state[i].leading_zeros()` matches the
/// leading zero bit count of the corresponding 32-bit slice of the hash.
#[inline(always)]
pub fn check_state_difficulty(state: &[u32; 8], difficulty: u32) -> bool {
    let req_bits = difficulty * 4;
    if req_bits <= 32 {
        state[0].leading_zeros() >= req_bits
    } else if req_bits <= 64 {
        state[0] == 0 && state[1].leading_zeros() >= req_bits - 32
    } else if req_bits <= 96 {
        state[0] == 0 && state[1] == 0 && state[2].leading_zeros() >= req_bits - 64
    } else {
        state[0] == 0
            && state[1] == 0
            && state[2] == 0
            && state[3].leading_zeros() >= req_bits - 96
    }
}

impl Default for Midstate {
    fn default() -> Self {
        Self::new()
    }
}

/// Write `n` as decimal into `out` (right-aligned, no leading zeros) and return
/// the number of digits. The digits are placed at `out[..len]`.
pub fn write_decimal(mut n: u64, out: &mut [u8; 20]) -> usize {
    let mut i = 20;
    if n == 0 {
        out[19] = b'0';
        out.copy_within(19..20, 0);
        return 1;
    }
    while n > 0 {
        i -= 1;
        out[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    let len = 20 - i;
    out.copy_within(i..20, 0);
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_bytes(data: &[u8]) -> [u8; 32] {
        let mut m = Midstate::new();
        m.update(data);
        m.finish(&[])
    }

    #[test]
    fn empty_string() {
        let h = hash_bytes(b"");
        assert_eq!(
            hex(h),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn abc() {
        let h = hash_bytes(b"abc");
        assert_eq!(
            hex(h),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn split_across_blocks() {
        // Tail appended via finish must equal hashing the whole thing at once.
        let prefix = b"task_123_diff_8_deadbeef_";
        let nonce = b"987654321";
        let mut whole = Midstate::new();
        let mut cat = prefix.to_vec();
        cat.extend_from_slice(nonce);
        whole.update(&cat);
        let expected = whole.finish(&[]);

        let mut m = Midstate::new();
        m.update(prefix);
        let got = m.finish(nonce);
        assert_eq!(got, expected);
    }

    #[test]
    fn decimal() {
        let mut b = [0u8; 20];
        let n = write_decimal(0, &mut b);
        assert_eq!(&b[..n], b"0");
        let n = write_decimal(42, &mut b);
        assert_eq!(&b[..n], b"42");
        let n = write_decimal(12345678901234, &mut b);
        assert_eq!(&b[..n], b"12345678901234");
    }

    #[test]
    fn multi_block() {
        // Exercise messages that span exactly one, two, and three 64-byte blocks.
        let h = |b: &[u8]| hex(hash_bytes(b));
        assert_eq!(
            h(&[b'a'; 64]),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
        );
        assert_eq!(
            h(&[b'a'; 65]),
            "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"
        );
        assert_eq!(
            h(&[b'a'; 140]),
            "c094ed2f614ab7a02e7557f8eba6b03b457ce7beacf1d8031088f97a1770e5e6"
        );
    }

    fn hex(h: [u8; 32]) -> String {
        crate::task::hex_lower(&h)
    }
}
