mod api;
mod backend;
mod orchestrator;
mod sha;
mod task;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use backend::Backend;
use task::{Id, Task};

pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[derive(Parser)]
#[command(
    name = "scam-miner",
    version,
    about = "Scummy Bank High-Performance Batch Miner (Rust)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List detected acceleration backends and exit
    #[command(name = "list-backends")]
    ListBackends {
        /// Number of CPU threads to report (default: max)
        #[arg(long)]
        cores: Option<usize>,
    },

    /// Start mining against the ledger
    Mine(MineArgs),

    /// Benchmark the selected acceleration backend(s) without a token
    Benchmark(BenchArgs),
}

#[derive(Args)]
struct BenchArgs {
    /// Acceleration backend to benchmark
    #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
    backend: BackendKind,

    /// Number of CPU threads (default: max)
    #[arg(long)]
    cores: Option<usize>,

    /// Benchmark duration in seconds
    #[arg(long, default_value_t = 5)]
    seconds: u64,

    /// Device index within the selected backend
    #[arg(long)]
    device: Option<usize>,
}

#[derive(Args)]
struct MineArgs {
    /// API Base URL
    #[arg(long, default_value = "https://scam.dn42")]
    url: String,

    /// User API token (required)
    #[arg(long, required = true)]
    token: String,

    /// Difficulty level (6-10)
    #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u32).range(6..=10))]
    difficulty: u32,

    /// Number of tasks to fetch and solve in one loop iteration
    #[arg(long, default_value_t = 100)]
    batch_size: u32,

    /// Number of CPU threads (default: max)
    #[arg(long)]
    cores: Option<usize>,

    /// Target IBAN (DN420042... or 16 digits) to receive payouts
    #[arg(long)]
    account: Option<String>,

    /// Acceleration backend to use
    #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
    backend: BackendKind,

    /// Device index within the selected backend
    #[arg(long)]
    device: Option<usize>,
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
enum BackendKind {
    Auto,
    Cpu,
    Vulkan,
    Opencl,
    Hip,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::ListBackends { cores } => {
            let cores = cores.unwrap_or_else(default_cores).max(1);
            print_backends(cores);
        }
        Command::Mine(args) => run_miner(args)?,
        Command::Benchmark(args) => run_benchmark(args)?,
    }
    Ok(())
}

fn run_miner(args: MineArgs) -> Result<()> {
    let cores = args
        .cores
        .unwrap_or_else(default_cores)
        .clamp(1, default_cores().max(1));

    let account = match &args.account {
        Some(a) => Some(parse_account(a)?),
        None => None,
    };

    let client = api::Client::new(args.url.clone(), args.token.clone())?;

    let backends = build_backends(args.backend, cores, args.device, false);

    ctrlc::set_handler(|| {
        SHUTDOWN.store(true, Ordering::Relaxed);
    })
    .ok();

    let orch = orchestrator::Orchestrator::new(
        client,
        backends,
        args.difficulty,
        args.batch_size,
        account,
    );
    orch.run()?;
    Ok(())
}

fn run_benchmark(args: BenchArgs) -> Result<()> {
    let cores = args
        .cores
        .unwrap_or_else(default_cores)
        .clamp(1, default_cores().max(1));

    let backends = build_backends(args.backend, cores, args.device, true);

    // Unsolvable task: difficulty 20 (80 zero bits) never matches, so each
    // `mine` call exhausts the full nonce range, measuring raw throughput.
    let task = Task {
        id: Id::Num(0),
        data: "benchmark_payload".into(),
        difficulty: 20,
        nonce_start: 0,
        nonce_end: 10_000_000 - 1,
    };

    println!("=== Benchmark ({}s per device) ===", args.seconds);
    for backend in &backends {
        let counter = AtomicU64::new(0);
        let start = Instant::now();
        let deadline = start + Duration::from_secs(args.seconds);
        loop {
            if Instant::now() >= deadline {
                break;
            }
            backend.mine(&task, &counter);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let hashes = counter.load(Ordering::Relaxed);
        let hps = hashes as f64 / elapsed;
        println!(
            "  {:<12} {} => {:>12.0} H/s  ({} hashes in {:.2}s)",
            backend.name(),
            backend.description(),
            hps,
            hashes,
            elapsed
        );
    }
    println!("{}", "=".repeat(40));
    Ok(())
}

#[cfg_attr(
    not(any(feature = "vulkan", feature = "opencl", feature = "hip")),
    allow(unused_variables)
)]
fn build_backends(
    kind: BackendKind,
    cores: usize,
    device: Option<usize>,
    all: bool,
) -> Vec<Arc<dyn Backend>> {
    let cpu: Arc<dyn Backend> = Arc::new(backend::cpu::CpuBackend::new(cores));

    #[cfg(feature = "vulkan")]
    let vulkan: Vec<Arc<dyn Backend>> = if all && device.is_none() {
        backend::vulkan::detect_all()
            .into_iter()
            .map(|v| Arc::new(v) as Arc<dyn Backend>)
            .collect()
    } else {
        backend::vulkan::detect(device)
            .into_iter()
            .map(|v| Arc::new(v) as Arc<dyn Backend>)
            .collect()
    };
    #[cfg(not(feature = "vulkan"))]
    let vulkan: Vec<Arc<dyn Backend>> = Vec::new();

    #[cfg(feature = "opencl")]
    let opencl: Vec<Arc<dyn Backend>> = if all && device.is_none() {
        backend::opencl::detect_all()
            .into_iter()
            .map(|o| Arc::new(o) as Arc<dyn Backend>)
            .collect()
    } else {
        backend::opencl::detect(device)
            .into_iter()
            .map(|o| Arc::new(o) as Arc<dyn Backend>)
            .collect()
    };
    #[cfg(not(feature = "opencl"))]
    let opencl: Vec<Arc<dyn Backend>> = Vec::new();

    #[cfg(feature = "hip")]
    let hip: Vec<Arc<dyn Backend>> = if all && device.is_none() {
        backend::hip::detect_all()
            .into_iter()
            .map(|h| Arc::new(h) as Arc<dyn Backend>)
            .collect()
    } else {
        backend::hip::detect(device)
            .into_iter()
            .map(|h| Arc::new(h) as Arc<dyn Backend>)
            .collect()
    };
    #[cfg(not(feature = "hip"))]
    let hip: Vec<Arc<dyn Backend>> = Vec::new();

    match kind {
        BackendKind::Cpu => vec![cpu],
        BackendKind::Vulkan => {
            if vulkan.is_empty() {
                eprintln!("[FATAL] Vulkan backend not available on this machine.");
                std::process::exit(1);
            }
            vulkan
        }
        BackendKind::Opencl => {
            if opencl.is_empty() {
                eprintln!("[FATAL] OpenCL backend not available on this machine.");
                std::process::exit(1);
            }
            opencl
        }
        BackendKind::Hip => {
            if hip.is_empty() {
                eprintln!("[FATAL] HIP backend not available on this machine.");
                std::process::exit(1);
            }
            hip
        }
        BackendKind::Auto => {
            let mut list = vulkan;
            list.extend(opencl);
            list.extend(hip);
            list.push(cpu);
            list
        }
    }
}

fn print_backends(cores: usize) {
    let mut rows: Vec<(String, String, String, String)> = Vec::new();

    rows.push((
        "cpu".into(),
        "-".into(),
        format!("sha256: {} ({} cores)", backend::cpu_sha_desc(), cores),
        "available".into(),
    ));

    #[cfg(feature = "vulkan")]
    push_device_rows(&mut rows, "vulkan", &backend::vulkan::list_devices());
    #[cfg(not(feature = "vulkan"))]
    rows.push((
        "vulkan".into(),
        "-".into(),
        "GPU compute".into(),
        "disabled".into(),
    ));

    #[cfg(feature = "opencl")]
    push_device_rows(&mut rows, "opencl", &backend::opencl::list_devices());
    #[cfg(not(feature = "opencl"))]
    rows.push((
        "opencl".into(),
        "-".into(),
        "GPU compute".into(),
        "disabled".into(),
    ));

    #[cfg(feature = "hip")]
    push_device_rows(&mut rows, "hip", &backend::hip::list_devices());
    #[cfg(not(feature = "hip"))]
    rows.push((
        "hip".into(),
        "-".into(),
        "GPU compute".into(),
        "disabled".into(),
    ));

    print_table(
        "Available acceleration backends:",
        &["BACKEND", "DEVICE", "DESCRIPTION", "STATUS"],
        &rows,
    );
}

/// Push one row per device of a backend, or a single `not available` row when
/// none were found.
#[cfg(any(feature = "vulkan", feature = "opencl", feature = "hip"))]
fn push_device_rows(rows: &mut Vec<(String, String, String, String)>, name: &str, devices: &[String]) {
    if devices.is_empty() {
        rows.push((
            name.into(),
            "-".into(),
            "GPU compute".into(),
            "not available".into(),
        ));
    } else {
        for (i, d) in devices.iter().enumerate() {
            rows.push((name.into(), i.to_string(), d.clone(), "available".into()));
        }
    }
}

/// Render an aligned text table from a header row and body rows.
fn print_table(title: &str, headers: &[&str; 4], rows: &[(String, String, String, String)]) {
    let width = |i: usize| -> usize {
        let body = rows.iter().map(|r| match i {
            0 => r.0.len(),
            1 => r.1.len(),
            2 => r.2.len(),
            _ => r.3.len(),
        });
        body.chain(std::iter::once(headers[i].len()))
            .max()
            .unwrap_or(0)
    };
    let (w0, w1, w2, w3) = (width(0), width(1), width(2), width(3));

    println!("{title}");
    println!(
        "  {:<w0$}  {:<w1$}  {:<w2$}  {:<w3$}",
        headers[0], headers[1], headers[2], headers[3]
    );
    println!(
        "  {:-<w0$}  {:-<w1$}  {:-<w2$}  {:-<w3$}",
        "", "", "", ""
    );
    for r in rows {
        println!(
            "  {:<w0$}  {:<w1$}  {:<w2$}  {:<w3$}",
            r.0, r.1, r.2, r.3
        );
    }
}

fn default_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn parse_account(account: &str) -> Result<String> {
    let clean: String = account
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .collect::<String>()
        .to_uppercase();

    let clean = if clean.len() == 16 && clean.chars().all(|c| c.is_ascii_digit()) {
        format!("DN420042{clean}")
    } else {
        clean
    };

    if !clean.starts_with("DN420042") || clean.len() != 24 {
        bail!("Invalid account format. Expected DN420042... (24 chars)");
    }
    Ok(clean)
}
