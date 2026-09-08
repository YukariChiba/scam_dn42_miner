use std::sync::atomic::AtomicU64;

use super::common::{
    build_tail_words, make_solution, mine_chunks, ChunkOutcome, Params, CONSTANT_WORDS, WORKGROUP,
};
use super::Backend;
use crate::sha::Midstate;
use crate::task::{Solution, Task};

pub struct VulkanBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    adapter_name: String,
    constant_buf: wgpu::Buffer,
    result_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
}

/// List the names of all Vulkan adapters available on this machine.
pub fn list_devices() -> Vec<String> {
    let instance = wgpu::Instance::default();
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .map(|a| a.get_info().name)
        .collect()
}

/// Probe for a Vulkan adapter (selected by `index`, defaulting to the first)
/// and build the pipeline. Returns `None` if no suitable adapter/device is
/// available (e.g. no GPU, no Vulkan driver, or the index is out of range).
pub fn detect(index: Option<usize>) -> Option<VulkanBackend> {
    let instance = wgpu::Instance::default();

    let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .nth(index.unwrap_or(0))?;

    from_adapter(adapter)
}

/// Build a backend for every Vulkan adapter available on this machine.
pub fn detect_all() -> Vec<VulkanBackend> {
    let instance = wgpu::Instance::default();
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .filter_map(from_adapter)
        .collect()
}

fn from_adapter(adapter: wgpu::Adapter) -> Option<VulkanBackend> {
    let adapter_name = adapter.get_info().name;

    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;

    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/sha256.wgsl"));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sha256 miner"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    // Allocated once and reused across every task/dispatch.
    let constant_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("constant_w"),
        size: (CONSTANT_WORDS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result"),
        size: 12,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: Params::SIZE as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 12,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: constant_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    Some(VulkanBackend {
        device,
        queue,
        pipeline,
        bind_group,
        adapter_name,
        constant_buf,
        result_buf,
        params_buf,
        readback_buf,
    })
}

impl VulkanBackend {
    fn readback_result(&self, src: &wgpu::Buffer) -> Option<(u64, u32)> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, 0, &self.readback_buf, 0, 12);
        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.readback_buf
            .map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send(r);
            });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok().and_then(|r| r.ok())?;

        let out = {
            let data = self.readback_buf.get_mapped_range(..).ok()?;
            let lo = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let hi = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            let flag = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
            let nonce = (lo as u64) | ((hi as u64) << 32);
            (nonce, flag)
        };
        self.readback_buf.unmap();
        Some(out)
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

        // Pre-compress the prefix into a midstate + partial tail block.
        let mut mid = Midstate::new();
        mid.update(prefix_bytes);
        let (state, partial_block, buffered) = mid.midstate_parts();
        let prefix_len = mid.total_len();

        // Reset the shared result buffer (result[2] flag must start at 0).
        self.queue
            .write_buffer(&self.result_buf, 0, bytemuck::bytes_of(&[0u32, 0u32, 0u32]));

        let nonce = mine_chunks(start, count, counter, |chunk_start, c, nd| {
            let (words, num_blocks) = build_tail_words(&partial_block, buffered, prefix_len, nd);
            let params = Params::new(
                chunk_start, c, task.difficulty, buffered, nd, num_blocks, state,
            );

            self.queue
                .write_buffer(&self.constant_buf, 0, bytemuck::cast_slice(&words));
            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(c.div_ceil(WORKGROUP), 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));

            match self.readback_result(&self.result_buf) {
                Some((nonce, flag)) if flag != 0 => ChunkOutcome::Found(nonce),
                Some(_) => ChunkOutcome::NotFound,
                None => ChunkOutcome::Abort,
            }
        })?;

        Some(make_solution(task, prefix_bytes, nonce))
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn description(&self) -> String {
        format!("Vulkan compute ({})", self.adapter_name)
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
    fn shader_parses() {
        let src = include_str!("../shaders/sha256.wgsl");
        match naga::front::wgsl::parse_str(src) {
            Ok(_) => {}
            Err(e) => panic!("WGSL failed to parse: {e}"),
        }
    }

    #[test]
    fn vulkan_finds_solution() {
        let Some(backend) = detect(None) else {
            eprintln!("no Vulkan adapter available; skipping test");
            return;
        };

        // Realistic multi-block prefix (like actual API tasks), nonce range
        // above u32::MAX to exercise the 64-bit nonce path.
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
