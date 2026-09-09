use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::api::{ApiError, Client};
use crate::backend::Backend;
use crate::task::{Solution, Task};

pub struct Orchestrator {
    client: Client,
    backends: Vec<Arc<dyn Backend>>,
    difficulty: u32,
    batch_size: u32,
    account: Option<String>,
    counter: Arc<AtomicU64>,
}

/// Thread-safe session statistics. Updated from worker threads as each
/// solution is found and submitted immediately (first-come, first-served).
struct Stats {
    tasks_completed: AtomicU64,
    races_lost: AtomicU64,
    total_earned: Mutex<f64>,
}

impl Stats {
    fn new() -> Self {
        Self {
            tasks_completed: AtomicU64::new(0),
            races_lost: AtomicU64::new(0),
            total_earned: Mutex::new(0.0),
        }
    }
}

impl Orchestrator {
    pub fn new(
        client: Client,
        backends: Vec<Arc<dyn Backend>>,
        difficulty: u32,
        batch_size: u32,
        account: Option<String>,
    ) -> Self {
        Self {
            client,
            backends,
            difficulty,
            batch_size,
            account,
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Distribute tasks among the available backends and mine them, running
    /// multiple backends (e.g. CPU + GPU) concurrently for throughput. Each
    /// solution is submitted immediately once found, to win the first-come,
    /// first-served race. Returns the number of solutions found.
    fn mine_tasks(&self, tasks: Vec<Task>, stats: &Stats) -> usize {
        let n = self.backends.len();
        if n == 1 {
            let backend = &self.backends[0];
            let mut found = 0usize;
            for task in &tasks {
                print_task(task);
                if let Some(sol) = backend.mine(task, self.counter.as_ref()) {
                    println!(
                        "\n    -> Solution found! Hash: {}... Nonce: {}",
                        &sol.hash[..16],
                        format_thousands(sol.nonce)
                    );
                    self.submit_solution(&sol, stats);
                    found += 1;
                } else {
                    println!("\n    -> Nonce range exhausted (no hit).");
                }
            }
            found
        } else {
            // Work-stealing: backends grab the next unsolved task from a shared
            // atomic index, keeping fast backends (GPU) saturated.
            let tasks = Arc::new(tasks);
            let task_idx = Arc::new(AtomicUsize::new(0));
            let found = Arc::new(AtomicUsize::new(0));

            std::thread::scope(|scope| {
                for backend in &self.backends {
                    let tasks = Arc::clone(&tasks);
                    let task_idx = Arc::clone(&task_idx);
                    let found = Arc::clone(&found);
                    let backend = Arc::clone(backend);

                    scope.spawn(move || {
                        loop {
                            let idx = task_idx.fetch_add(1, Ordering::Relaxed);
                            if idx >= tasks.len() || crate::SHUTDOWN.load(Ordering::Relaxed) {
                                break;
                            }
                            let task = &tasks[idx];
                            print_task(task);
                            if let Some(sol) = backend.mine(task, self.counter.as_ref()) {
                                println!(
                                    "\n    -> [{}] Solution found! Hash: {}... Nonce: {}",
                                    backend.name(),
                                    &sol.hash[..16],
                                    format_thousands(sol.nonce)
                                );
                                self.submit_solution(&sol, stats);
                                found.fetch_add(1, Ordering::Relaxed);
                            } else {
                                println!("\n    -> [{}] Nonce range exhausted (no hit).", backend.name());
                            }
                        }
                    });
                }
            });

            found.load(Ordering::Relaxed)
        }
    }

    /// Submit a single freshly-found solution to the ledger immediately and
    /// record the outcome in the shared session stats.
    fn submit_solution(&self, sol: &Solution, stats: &Stats) {
        let batch = [sol.clone()];
        match self.client.submit(&batch, self.account.as_deref()) {
            Ok(resp) => {
                if resp.accepted > 0 {
                    let earned: f64 = resp
                        .results
                        .iter()
                        .filter(|r| r.status == "solved")
                        .map(|r| r.reward)
                        .sum();
                    stats
                        .tasks_completed
                        .fetch_add(resp.accepted, Ordering::Relaxed);
                    *stats.total_earned.lock().unwrap() += earned;
                    println!("    -> Broadcasting to ledger immediately... [CLAIMED! +{earned:.3} FTK]");
                    if resp.consolidated > 0 {
                        println!(
                            "    ➔ [BLOCKCHAIN] {} blocks consolidated! New balance: {:.2} FTK",
                            format_thousands(resp.consolidated),
                            resp.new_balance
                        );
                    }
                } else {
                    let reason = resp
                        .results
                        .iter()
                        .filter(|r| r.status != "solved")
                        .filter_map(|r| r.error.clone())
                        .next()
                        .unwrap_or_else(|| "Another miner submitted first!".to_string());
                    if reason.to_lowercase().contains("rate limit") {
                        println!("    -> Broadcasting to ledger immediately... [RATE-LIMITED] {reason}");
                    } else {
                        stats.races_lost.fetch_add(1, Ordering::Relaxed);
                        println!("    -> Broadcasting to ledger immediately... [LOST RACE] {reason}");
                    }
                }
            }
            Err(e) => {
                println!("    -> Broadcasting to ledger immediately... [REJECTED] {e}");
            }
        }
    }

    pub fn run(&self) -> Result<()> {
        println!("=== Scummybank Miner ===");
        println!("API Host            : {}", self.client.base());
        println!("Target Difficulty   : Level {}", self.difficulty);
        println!("Fetch Chunk Size    : {} ranges", self.batch_size);
        for b in &self.backends {
            println!("Backend             : {}", b.description());
        }
        if let Some(a) = &self.account {
            println!("Target Account      : {}", a);
        } else {
            println!("Target Account      : Default (Primary Account)");
        }
        println!("{}", "-".repeat(55));

        let stats = Stats::new();

        loop {
            if crate::SHUTDOWN.load(Ordering::Relaxed) {
                break;
            }

            let result = self.step(&stats);
            match result {
                Ok(()) => {}
                Err(e) => {
                    if crate::SHUTDOWN.load(Ordering::Relaxed) {
                        break;
                    }
                    println!("\n[ERROR] Unexpected error: {e}. Retrying in 3 seconds...");
                    sleep_interruptible(Duration::from_secs(3));
                }
            }
        }

        if crate::SHUTDOWN.load(Ordering::Relaxed) {
            println!("\n[STOP] Miner stopped by user.");
        }

        let hashes = self.counter.load(Ordering::Relaxed);
        println!("\n{} Session Stats {}", "=".repeat(22), "=".repeat(22));
        println!(
            "Blocks Successfully Claimed : {}",
            stats.tasks_completed.load(Ordering::Relaxed)
        );
        println!(
            "Races Lost (Too Slow)       : {}",
            stats.races_lost.load(Ordering::Relaxed)
        );
        println!(
            "Total Earned (Est.)         : {:.3} FTK",
            *stats.total_earned.lock().unwrap()
        );
        println!("Total Hashes Computed       : {}", format_thousands(hashes));
        println!("{}", "=".repeat(59));
        Ok(())
    }

    fn step(&self, stats: &Stats) -> Result<()> {
        println!("[*] Requesting {} tasks from ledger...", self.batch_size);
        let tasks = match self.client.get_tasks(self.difficulty, self.batch_size) {
            Ok(t) => t,
            Err(ApiError::RateLimited(err)) => {
                println!("\n[RATE-LIMIT]{err}");
                println!("[ADVICE] Sleeping for 3s... (Consider switching to --difficulty 8+ to bypass limits)");
                sleep_interruptible(Duration::from_secs(3));
                return Ok(());
            }
            Err(ApiError::Unauthorized) => {
                println!("\n[FATAL] Unauthorized. Check your --token parameter.");
                std::process::exit(1);
            }
            Err(ApiError::Other(e)) => return Err(anyhow::anyhow!(e)),
        };

        if tasks.is_empty() {
            sleep_interruptible(Duration::from_secs(3));
            return Ok(());
        }

        println!("[+] Loaded {} tasks. Starting calculation loop...", tasks.len());

        let started = Instant::now();
        let base_hashes = self.counter.load(Ordering::Relaxed);
        let found = self.mine_tasks(tasks, stats);
        let elapsed = started.elapsed().as_secs_f64();
        let delta = self.counter.load(Ordering::Relaxed) - base_hashes;
        if elapsed > 0.0 {
            println!(
                "[*] Batch hashrate: {:.0} H/s",
                delta as f64 / elapsed
            );
        }

        if found == 0 {
            println!("[-] Done with batch, but zero solutions found.");
        }
        println!("{}", "=".repeat(55));
        Ok(())
    }
}

fn print_task(task: &Task) {
    let now = chrono_now();
    println!(
        "[{now}] Mining Block #{} (Diff: {}) Range: [{} - {}]",
        task.id,
        task.difficulty,
        format_thousands(task.nonce_start),
        format_thousands(task.nonce_end)
    );
}

/// Format an integer with thousands separators (matches Python's `{:,}`).
fn format_thousands(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

fn chrono_now() -> String {
    // Avoid a full chrono dependency for a simple HH:MM:SS timestamp.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

fn sleep_interruptible(d: Duration) {
    let step = Duration::from_millis(100);
    let mut slept = Duration::ZERO;
    while slept < d {
        if crate::SHUTDOWN.load(Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(step);
        slept += step;
    }
}
