use std::ptr;
use std::sync::atomic::AtomicU64;
use std::sync::Mutex;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;

use super::common::{
    build_tail_words, make_solution, mine_chunks, ChunkOutcome, Params, CONSTANT_WORDS, WORKGROUP,
};
use super::Backend;
use crate::sha::Midstate;
use crate::task::{Solution, Task};

/// `opencl3::Kernel` is `Send` but not `Sync`. OpenCL objects are
/// reference-counted and thread-safe, so sharing the kernel immutably across
/// threads is sound; this newtype opts into `Sync` for that reason.
struct SyncKernel(Kernel);
unsafe impl Sync for SyncKernel {}

pub struct OpenclBackend {
    // Kept alive for the lifetime of the buffers/kernel created from it.
    _context: Context,
    queue: CommandQueue,
    kernel: SyncKernel,
    device_name: String,
    // Pre-allocated buffers, reused across every task/dispatch. `Buffer` is
    // `Sync`, but the write API takes `&mut`, so they're guarded by a `Mutex`.
    constant_buf: Mutex<Buffer<u32>>,
    result_buf: Mutex<Buffer<u32>>,
    params_buf: Mutex<Buffer<Params>>,
}

fn is_amd(device_id: opencl3::types::cl_device_id) -> bool {
    let d = Device::new(device_id);
    let vendor = d.vendor().unwrap_or_default();
    let name = d.name().unwrap_or_default();
    vendor.to_lowercase().contains("advanced micro")
        || vendor.to_lowercase().contains("amd")
        || name.to_lowercase().contains("amd")
        || name.to_lowercase().contains("radeon")
        || name.to_lowercase().contains("instinct")
}

/// List the names of all GPU OpenCL devices available on this machine.
pub fn list_devices() -> Vec<String> {
    let Ok(device_ids) = get_all_devices(CL_DEVICE_TYPE_GPU) else {
        return Vec::new();
    };
    device_ids
        .iter()
        .copied()
        .map(|id| Device::new(id).name().unwrap_or_else(|_| "unknown".into()))
        .collect()
}

/// Probe for an OpenCL device and build the kernel. When `index` is `Some`,
/// select that device; otherwise prefer an AMD device (ROCm) and fall back to
/// the first available GPU. Returns `None` if no OpenCL platform/device is
/// available or the index is out of range.
pub fn detect(index: Option<usize>) -> Option<OpenclBackend> {
    let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU).ok()?;
    if device_ids.is_empty() {
        return None;
    }

    // Prefer an AMD device (ROCm), fall back to the first available GPU.
    let device_id = match index {
        Some(i) => *device_ids.get(i)?,
        None => device_ids
            .iter()
            .copied()
            .find(|&id| is_amd(id))
            .unwrap_or(device_ids[0]),
    };

    build_from_device(device_id)
}

/// Build a backend for every GPU OpenCL device available on this machine.
pub fn detect_all() -> Vec<OpenclBackend> {
    let Ok(device_ids) = get_all_devices(CL_DEVICE_TYPE_GPU) else {
        return Vec::new();
    };
    device_ids
        .into_iter()
        .filter_map(build_from_device)
        .collect()
}

fn build_from_device(device_id: opencl3::types::cl_device_id) -> Option<OpenclBackend> {
    let device = Device::new(device_id);
    let device_name = device.name().unwrap_or_else(|_| "unknown".into());

    let context = Context::from_device(&device).ok()?;
    let queue = CommandQueue::create_default(&context, 0).ok()?;

    let program = Program::create_and_build_from_source(
        &context,
        include_str!("../shaders/sha256.cl"),
        "",
    )
    .ok()?;
    let kernel = Kernel::create(&program, "mine").ok()?;

    let constant_buf = unsafe {
        Buffer::<u32>::create(&context, CL_MEM_READ_ONLY, CONSTANT_WORDS, ptr::null_mut()).ok()?
    };
    let result_buf = unsafe {
        Buffer::<u32>::create(&context, CL_MEM_READ_WRITE, 3, ptr::null_mut()).ok()?
    };
    let params_buf = unsafe {
        Buffer::<Params>::create(&context, CL_MEM_READ_ONLY, 1, ptr::null_mut()).ok()?
    };

    Some(OpenclBackend {
        _context: context,
        queue,
        kernel: SyncKernel(kernel),
        device_name,
        constant_buf: Mutex::new(constant_buf),
        result_buf: Mutex::new(result_buf),
        params_buf: Mutex::new(params_buf),
    })
}

impl OpenclBackend {
    fn readback_result(&self) -> Option<(u64, u32)> {
        let mut out = [0u32; 3];
        unsafe {
            let result_buf = self.result_buf.lock().unwrap();
            self.queue
                .enqueue_read_buffer(&result_buf, CL_BLOCKING, 0, &mut out, &[])
                .ok()?;
        }
        let nonce = (out[0] as u64) | ((out[1] as u64) << 32);
        Some((nonce, out[2]))
    }

    fn mine(&self, task: &Task, counter: &AtomicU64) -> Option<Solution> {
        let start = task.nonce_start;
        let end = task.nonce_end;
        if end < start {
            return None;
        }
        let count = end - start + 1;
        if count == 0 {
            return None;
        }

        let prefix = task.prefix();
        let prefix_bytes = prefix.as_bytes();

        let mut mid = Midstate::new();
        mid.update(prefix_bytes);
        let (state, partial_block, buffered) = mid.midstate_parts();
        let prefix_len = mid.total_len();

        // Reset the result buffer (result[2] flag must start at 0).
        unsafe {
            let mut result_buf = self.result_buf.lock().unwrap();
            self.queue
                .enqueue_write_buffer(&mut result_buf, CL_BLOCKING, 0, &[0u32, 0, 0], &[])
                .ok()?;
        }

        let nonce = mine_chunks(start, count, counter, |chunk_start, c, nd| {
            let (words, num_blocks) = build_tail_words(&partial_block, buffered, prefix_len, nd);
            let params = Params::new(
                chunk_start, c, task.difficulty, buffered, nd, num_blocks, state,
            );

            let mut constant_buf = self.constant_buf.lock().unwrap();
            let mut params_buf = self.params_buf.lock().unwrap();

            if unsafe {
                self.queue
                    .enqueue_write_buffer(&mut *constant_buf, CL_BLOCKING, 0, &words, &[])
            }
            .is_err()
                || unsafe {
                    self.queue
                        .enqueue_write_buffer(&mut *params_buf, CL_BLOCKING, 0, &[params], &[])
                }
                .is_err()
            {
                return ChunkOutcome::Abort;
            }

            let global = (c.div_ceil(WORKGROUP) * WORKGROUP) as usize;
            {
                let result_buf = self.result_buf.lock().unwrap();
                if unsafe {
                    ExecuteKernel::new(&self.kernel.0)
                        .set_arg(&*constant_buf)
                        .set_arg(&*result_buf)
                        .set_arg(&*params_buf)
                        .set_global_work_size(global)
                        .set_local_work_size(WORKGROUP as usize)
                        .enqueue_nd_range(&self.queue)
                }
                .is_err()
                {
                    return ChunkOutcome::Abort;
                }
            }

            match self.readback_result() {
                Some((nonce, flag)) if flag != 0 => ChunkOutcome::Found(nonce),
                Some(_) => ChunkOutcome::NotFound,
                None => ChunkOutcome::Abort,
            }
        })?;

        Some(make_solution(task, prefix_bytes, nonce))
    }
}

impl Backend for OpenclBackend {
    fn name(&self) -> &'static str {
        "opencl"
    }

    fn description(&self) -> String {
        format!("OpenCL ({})", self.device_name)
    }

    fn mine(&self, task: &Task, counter: &AtomicU64) -> Option<Solution> {
        self.mine(task, counter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha::write_decimal;
    use crate::task::{check_difficulty, Id};

    #[test]
    fn kernel_source_sanity() {
        let src = include_str!("../shaders/sha256.cl");
        assert!(src.contains("__kernel void mine"));
        assert!(src.contains("atomic_cmpxchg"));
    }

    #[test]
    fn opencl_finds_solution() {
        let Some(backend) = detect(None) else {
            eprintln!("no OpenCL device available; skipping test");
            return;
        };

        let data = "{\"difficulty\": 6, \"random\": \"406092c935b83ad510951faf1dfadcb6\", \"timestamp\": \"2026-09-08T12:34:44.166997\", \"user_id\": 16}";
        let task = Task {
            id: Id::Num(60346),
            data: data.into(),
            difficulty: 2,
            nonce_start: 5_000_000_000,
            nonce_end: 5_010_000_000,
        };
        let counter = AtomicU64::new(0);
        let sol = backend.mine(&task, &counter).expect("should find a solution");
        assert!(sol.nonce >= 5_000_000_000);

        let mut mid = Midstate::new();
        mid.update(task.prefix().as_bytes());
        let mut d = [0u8; 20];
        let n = write_decimal(sol.nonce, &mut d);
        let h = mid.finish(&d[..n]);
        assert!(check_difficulty(&h, task.difficulty));
    }
}
