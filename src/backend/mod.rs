use std::sync::atomic::AtomicU64;

use crate::task::{Solution, Task};

#[cfg(any(feature = "vulkan", feature = "opencl", feature = "hip"))]
pub mod common;
pub mod cpu;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "hip")]
pub mod hip;

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> String;
    fn mine(&self, task: &Task, counter: &AtomicU64) -> Option<Solution>;
}

/// Human-readable description of the active CPU SHA-256 acceleration path.
pub fn cpu_sha_desc() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sha")
            && std::arch::is_x86_feature_detected!("sse4.1")
        {
            return "SHA-NI";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("sha2") {
            return "ARMv8 Crypto";
        }
    }

    "software"
}
