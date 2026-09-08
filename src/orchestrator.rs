use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::api::Client;
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
    /// multiple backends (e.g. CPU + GPU) concurrently for throughput.
    fn mine_tasks(&self, tasks: Vec<Task>) -> Vec<Solution> {
        let n = self.backends.len();
        if n == 1 {
            let backend = &self.backends[0];
            let mut sols = Vec::new();
            for task in &tasks {
                print_task(task);
                if let Some(sol) = backend.mine(task, self.counter.as_ref()) {
                    println!("\n    -> Solved! Nonce: {}", sol.nonce);
                    sols.push(sol);
                } else {
                    println!("\n    -> Failed to solve (nonce exhausted)");
                }
            }
            sols
        } else {
            // Work-stealing: backends grab the next unsolved task from a shared
            // atomic index, keeping fast backends (GPU) saturated.
            let tasks = Arc::new(tasks);
            let task_idx = Arc::new(AtomicUsize::new(0));
            let solutions = Arc::new(Mutex::new(Vec::new()));

            std::thread::scope(|scope| {
                for backend in &self.backends {
                    let tasks = Arc::clone(&tasks);
                    let task_idx = Arc::clone(&task_idx);
                    let solutions = Arc::clone(&solutions);
                    let counter = Arc::clone(&self.counter);
                    let backend = Arc::clone(backend);

                    scope.spawn(move || {
                        loop {
                            let idx = task_idx.fetch_add(1, Ordering::Relaxed);
                            if idx >= tasks.len() || crate::SHUTDOWN.load(Ordering::Relaxed) {
                                break;
                            }
                            let task = &tasks[idx];
                            print_task(task);
                            if let Some(sol) = backend.mine(task, counter.as_ref()) {
                                println!(
                                    "\n    -> [{}] Solved! Nonce: {}",
                                    backend.name(),
                                    sol.nonce
                                );
                                solutions.lock().unwrap().push(sol);
                            } else {
                                println!("\n    -> [{}] Failed (nonce exhausted)", backend.name());
                            }
                        }
                    });
                }
            });

            let res = solutions.lock().unwrap().clone();
            res
        }
    }

    pub fn run(&self) -> Result<()> {
        println!("=== Scummy Bank High-Performance Batch Miner (Rust) ===");
        println!("Difficulty    : Level {}", self.difficulty);
        println!("Batch Size    : {} tasks", self.batch_size);
        for b in &self.backends {
            println!("Backend       : {}", b.description());
        }
        if let Some(a) = &self.account {
            println!("Target IBAN   : {}", a);
        } else {
            println!("Target IBAN   : Default (Primary Account)");
        }
        println!("{}", "-".repeat(55));

        let mut tasks_completed: u64 = 0;
        let mut total_earned: f64 = 0.0;

        loop {
            if crate::SHUTDOWN.load(Ordering::Relaxed) {
                break;
            }

            let result = self.step(&mut tasks_completed, &mut total_earned);
            match result {
                Ok(()) => {}
                Err(e) => {
                    if crate::SHUTDOWN.load(Ordering::Relaxed) {
                        break;
                    }
                    eprintln!("\n[ERROR] {e}. Re-trying in 5 seconds...");
                    sleep_interruptible(Duration::from_secs(5));
                }
            }
        }

        let hashes = self.counter.load(Ordering::Relaxed);
        println!("\n{} Session Stats {}", "=".repeat(22), "=".repeat(22));
        println!("Total Tasks Solved & Submitted : {tasks_completed}");
        println!("Total Hashes Computed          : {hashes}");
        println!("{}", "=".repeat(59));
        Ok(())
    }

    fn step(&self, tasks_completed: &mut u64, total_earned: &mut f64) -> Result<()> {
        println!("[*] Requesting {} tasks from ledger...", self.batch_size);
        let tasks = self.client.get_tasks(self.difficulty, self.batch_size)?;
        if tasks.is_empty() {
            println!("[-] No tasks received. Sleeping for 5s...");
            sleep_interruptible(Duration::from_secs(5));
            return Ok(());
        }

        println!("[+] Loaded {} tasks. Starting calculation loop...", tasks.len());

        let started = Instant::now();
        let base_hashes = self.counter.load(Ordering::Relaxed);
        let solutions = self.mine_tasks(tasks);
        let elapsed = started.elapsed().as_secs_f64();
        let delta = self.counter.load(Ordering::Relaxed) - base_hashes;
        if elapsed > 0.0 {
            println!(
                "[*] Batch hashrate: {:.0} H/s",
                delta as f64 / elapsed
            );
        }

        if solutions.is_empty() {
            println!("[-] Done with batch, but zero solutions found.");
            return Ok(());
        }

        println!("[*] Submitting batch of {} solved tasks to ledger...", solutions.len());
        match self.client.submit(&solutions, self.account.as_deref()) {
            Ok(resp) => {
                println!(
                    "[SUCCESS] Server accepted {}/{} solutions.",
                    resp.accepted,
                    solutions.len()
                );
                if resp.consolidated > 0 {
                    println!(
                        "[BLOCKCHAIN] {} blocks consolidated on disk! New balance: {:.2} Funtiks",
                        resp.consolidated, resp.new_balance
                    );
                } else {
                    println!("[ACCUMULATION] Rewards reserved! (Next consolidation at 1000 blocks).");
                }
                *tasks_completed += resp.accepted;
                *total_earned += resp.new_balance;
            }
            Err(e) => {
                println!("[REJECTED] Batch submission failed: {e}");
            }
        }
        println!("{}", "=".repeat(55));
        Ok(())
    }
}

fn print_task(task: &Task) {
    let now = chrono_now();
    println!(
        "[{now}] Processing task #{} (diff: {})",
        task.id, task.difficulty
    );
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
