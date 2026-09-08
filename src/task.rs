use serde::{Deserialize, Serialize};
use std::fmt;

/// A task id as returned by the API. The server may use a number or a string,
/// so accept both and echo it back verbatim on submission.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum Id {
    Num(u64),
    Str(String),
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Id::Num(n) => write!(f, "{n}"),
            Id::Str(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct Task {
    pub id: Id,
    pub data: String,
    pub difficulty: u32,
    pub nonce_start: u64,
    pub nonce_end: u64,
}

impl Task {
    /// The fixed prefix hashed for every nonce in this task, matching miner.py:
    /// `f"task_{task_id}_diff_{difficulty}_{raw_data}_"`.
    pub fn prefix(&self) -> String {
        format!("task_{}_diff_{}_{}_", self.id, self.difficulty, self.data)
    }
}

#[derive(Serialize, Clone)]
pub struct Solution {
    pub task_id: Id,
    pub nonce: u64,
    pub hash: String,
}

/// Check whether `hash` has at least `difficulty` leading zero hex nibbles.
/// Used only by tests to independently verify solutions produced by the
/// optimized `check_state_difficulty` path.
#[cfg(test)]
pub fn check_difficulty(hash: &[u8; 32], difficulty: u32) -> bool {
    let bits = (difficulty * 4) as usize;
    let full = bits / 8;
    let rem = bits % 8;
    if hash[..full].iter().any(|&b| b != 0) {
        return false;
    }
    if rem > 0 && hash[full] >> (8 - rem) != 0 {
        return false;
    }
    true
}

/// Lowercase hex encoding of a byte slice (matches Python `hexdigest()`).
pub fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difficulty_check() {
        // 24 leading zero bits = 6 hex zeros
        let mut h = [0u8; 32];
        h[0] = 0x0f; // top nibble zero, next nibble f
        assert!(!check_difficulty(&h, 6)); // 6 nibbles = 24 bits, we have only 4 zero bits
        assert!(check_difficulty(&h, 1));

        let mut h2 = [0u8; 32];
        h2[0] = 0x00;
        h2[1] = 0x00;
        h2[2] = 0x0f; // 2 full bytes + top nibble = 20 zero bits
        assert!(check_difficulty(&h2, 5)); // 20 bits
        assert!(!check_difficulty(&h2, 6)); // 24 bits needed
    }

    #[test]
    fn id_display() {
        assert_eq!(Id::Num(42).to_string(), "42");
        assert_eq!(Id::Str("abc".into()).to_string(), "abc");
    }
}
