use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use super::{cpu_sha_desc, Backend};
use crate::sha::{check_state_difficulty, state_to_bytes, write_decimal, Midstate};
use crate::task::{hex_lower, Solution, Task};

const BATCH: u64 = 256;
const FLUSH: u64 = 4096;

/// Mine a single task across `cores` threads using work-stealing over the
/// nonce range. Returns `(nonce, hash)` on success.
pub fn mine(task: &Task, cores: usize, counter: &AtomicU64) -> Option<(u64, [u8; 32])> {
    let prefix = task.prefix();
    let mut mid = Midstate::new();
    mid.update(prefix.as_bytes());

    let diff = task.difficulty;
    let start = task.nonce_start;
    let end = task.nonce_end;
    if end < start {
        return None;
    }

    let next = AtomicU64::new(start);
    let stop = AtomicBool::new(false);
    let found: Mutex<Option<(u64, [u8; 32])>> = Mutex::new(None);

    std::thread::scope(|s| {
        for _ in 0..cores {
            s.spawn(|| {
                let mut local = 0u64;
                let mut digits = [0u8; 20];
                loop {
                    if stop.load(Ordering::Relaxed) || crate::SHUTDOWN.load(Ordering::Relaxed) {
                        break;
                    }
                    let base = next.fetch_add(BATCH, Ordering::Relaxed);
                    if base > end {
                        break;
                    }
                    let hi = (base + BATCH - 1).min(end);
                    let mut n_len = write_decimal(base, &mut digits);
                    for nonce in base..=hi {
                        if nonce > base {
                            // Increment the decimal string in place via carry
                            // propagation instead of a full divide-by-10 pass.
                            let mut idx = n_len - 1;
                            loop {
                                if digits[idx] < b'9' {
                                    digits[idx] += 1;
                                    break;
                                }
                                digits[idx] = b'0';
                                if idx == 0 {
                                    digits.copy_within(0..n_len, 1);
                                    digits[0] = b'1';
                                    n_len += 1;
                                    break;
                                }
                                idx -= 1;
                            }
                        }

                        let state = mid.finish_raw(&digits[..n_len]);
                        local += 1;
                        if check_state_difficulty(&state, diff) {
                            let h = state_to_bytes(&state);
                            *found.lock().unwrap() = Some((nonce, h));
                            stop.store(true, Ordering::Relaxed);
                            counter.fetch_add(local, Ordering::Relaxed);
                            return;
                        }
                    }
                    if local >= FLUSH {
                        counter.fetch_add(local, Ordering::Relaxed);
                        local = 0;
                    }
                }
                counter.fetch_add(local, Ordering::Relaxed);
            });
        }
    });

    let result = found.lock().unwrap().take();
    result
}

pub struct CpuBackend {
    cores: usize,
}

impl CpuBackend {
    pub fn new(cores: usize) -> Self {
        Self {
            cores: cores.max(1),
        }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn description(&self) -> String {
        format!("CPU ({} cores, sha256: {})", self.cores, cpu_sha_desc())
    }

    fn mine(&self, task: &Task, counter: &AtomicU64) -> Option<Solution> {
        mine(task, self.cores, counter).map(|(nonce, hash)| Solution {
            task_id: task.id.clone(),
            nonce,
            hash: hex_lower(&hash),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{check_difficulty, Id};

    #[test]
    fn cpu_finds_solution() {
        let task = Task {
            id: Id::Num(1),
            data: "deadbeef".into(),
            difficulty: 2,
            nonce_start: 0,
            nonce_end: 100_000,
        };
        let counter = AtomicU64::new(0);
        let (nonce, hash) = mine(&task, 4, &counter).expect("should find a solution");
        assert!(check_difficulty(&hash, task.difficulty));

        let mut mid = Midstate::new();
        mid.update(task.prefix().as_bytes());
        let mut d = [0u8; 20];
        let n = write_decimal(nonce, &mut d);
        assert_eq!(mid.finish(&d[..n]), hash);
    }

    #[test]
    #[ignore]
    fn bench_hashrate() {
        use std::time::Instant;
        let task = Task {
            id: Id::Num(1),
            data: "deadbeef".into(),
            difficulty: 10, // won't be found in the range below
            nonce_start: 0,
            nonce_end: 10_000_000,
        };
        let counter = AtomicU64::new(0);
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let start = Instant::now();
        let r = mine(&task, cores, &counter);
        let elapsed = start.elapsed().as_secs_f64();
        assert!(r.is_none());
        let hashes = counter.load(Ordering::Relaxed);
        eprintln!(
            "hashed {hashes} in {elapsed:.3}s on {cores} cores = {:.0} H/s",
            hashes as f64 / elapsed
        );
    }
}
