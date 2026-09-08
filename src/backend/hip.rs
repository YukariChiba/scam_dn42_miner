use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;
use std::sync::atomic::AtomicU64;

use libloading::Library;

use super::common::{
    build_tail_words, make_solution, mine_chunks, ChunkOutcome, Params, CONSTANT_WORDS, WORKGROUP,
};
use super::Backend;
use crate::sha::Midstate;
use crate::task::{Solution, Task};

// HIP runtime / hiprtc types (matching the ROCm C headers).
type HipError = c_int;
type HiprtcResult = c_int;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HiprtcProgram = *mut c_void;

type FnGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type FnDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
type FnGetDeviceProperties = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type FnMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type FnMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int;
type FnModuleLoadData = unsafe extern "C" fn(*mut HipModule, *const c_void) -> c_int;
type FnModuleGetFunction = unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> c_int;
type FnLaunchKernel = unsafe extern "C" fn(
    HipFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;
type FnFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnModuleUnload = unsafe extern "C" fn(HipModule) -> c_int;
type FnGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

type FnRtcCreateProgram = unsafe extern "C" fn(
    *mut HiprtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type FnRtcCompileProgram = unsafe extern "C" fn(HiprtcProgram, c_int, *const *const c_char) -> c_int;
type FnRtcGetCodeSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> c_int;
type FnRtcGetCode = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> c_int;
type FnRtcGetLogSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> c_int;
type FnRtcGetLog = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> c_int;
type FnRtcDestroyProgram = unsafe extern "C" fn(*mut HiprtcProgram) -> c_int;

const HIP_SUCCESS: HipError = 0;
const HIPRTC_SUCCESS: HiprtcResult = 0;

/// Loaded HIP runtime + hiprtc entry points. Keeps the shared libraries alive
/// for the process lifetime. `Library` is `Send + Sync`, and `unsafe extern
/// "C" fn` pointers are too, so this is safe to share across threads.
struct HipLib {
    _runtime: Library,
    _rtc: Library,
    hip_get_device_count: FnGetDeviceCount,
    hip_device_get_name: FnDeviceGetName,
    hip_get_device_properties: FnGetDeviceProperties,
    hip_malloc: FnMalloc,
    hip_memcpy_htod: FnMemcpy,
    hip_memcpy_dtoh: FnMemcpy,
    hip_module_load_data: FnModuleLoadData,
    hip_module_get_function: FnModuleGetFunction,
    hip_module_launch_kernel: FnLaunchKernel,
    hip_free: FnFree,
    hip_module_unload: FnModuleUnload,
    hip_get_error_string: FnGetErrorString,
    hiprtc_create_program: FnRtcCreateProgram,
    hiprtc_compile_program: FnRtcCompileProgram,
    hiprtc_get_code_size: FnRtcGetCodeSize,
    hiprtc_get_code: FnRtcGetCode,
    hiprtc_get_log_size: FnRtcGetLogSize,
    hiprtc_get_log: FnRtcGetLog,
    hiprtc_destroy_program: FnRtcDestroyProgram,
}

impl HipLib {
    unsafe fn err_str(&self, code: c_int) -> String {
        unsafe {
            let p = (self.hip_get_error_string)(code);
            if p.is_null() {
                format!("HIP error {code}")
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }
}

const RESULT_SIZE: usize = 3 * 4; // lo, hi, flag
const CONSTANT_SIZE: usize = CONSTANT_WORDS * 4; // 2 tail blocks of 16 words

unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
    let s: libloading::Symbol<'_, T> = lib.get(name).ok()?;
    Some(*s)
}

fn load_library(names: &[&str]) -> Option<Library> {
    names
        .iter()
        .find_map(|n| unsafe { Library::new(*n).ok() })
}

fn load_hip_lib() -> Option<HipLib> {
    let runtime = load_library(&[
        "libamdhip64.so",
        "libamdhip64.so.6",
        "libamdhip64.so.5",
        "libamdhip64.so.7",
        "/opt/rocm/lib/libamdhip64.so",
    ])?;
    let rtc = load_library(&[
        "libhiprtc.so",
        "libhiprtc.so.6",
        "libhiprtc.so.5",
        "libhiprtc.so.7",
        "/opt/rocm/lib/libhiprtc.so",
    ])?;

    unsafe {
        Some(HipLib {
            hip_get_device_count: sym(&runtime, b"hipGetDeviceCount\0")?,
            hip_device_get_name: sym(&runtime, b"hipDeviceGetName\0")?,
            hip_get_device_properties: sym(&runtime, b"hipGetDeviceProperties\0")?,
            hip_malloc: sym(&runtime, b"hipMalloc\0")?,
            hip_memcpy_htod: sym(&runtime, b"hipMemcpyHtoD\0")?,
            hip_memcpy_dtoh: sym(&runtime, b"hipMemcpyDtoH\0")?,
            hip_module_load_data: sym(&runtime, b"hipModuleLoadData\0")?,
            hip_module_get_function: sym(&runtime, b"hipModuleGetFunction\0")?,
            hip_module_launch_kernel: sym(&runtime, b"hipModuleLaunchKernel\0")?,
            hip_free: sym(&runtime, b"hipFree\0")?,
            hip_module_unload: sym(&runtime, b"hipModuleUnload\0")?,
            hip_get_error_string: sym(&runtime, b"hipGetErrorString\0")?,
            hiprtc_create_program: sym(&rtc, b"hiprtcCreateProgram\0")?,
            hiprtc_compile_program: sym(&rtc, b"hiprtcCompileProgram\0")?,
            hiprtc_get_code_size: sym(&rtc, b"hiprtcGetCodeSize\0")?,
            hiprtc_get_code: sym(&rtc, b"hiprtcGetCode\0")?,
            hiprtc_get_log_size: sym(&rtc, b"hiprtcGetProgramLogSize\0")?,
            hiprtc_get_log: sym(&rtc, b"hiprtcGetProgramLog\0")?,
            hiprtc_destroy_program: sym(&rtc, b"hiprtcDestroyProgram\0")?,
            _runtime: runtime,
            _rtc: rtc,
        })
    }
}

/// Read the device's GCN architecture name (e.g. "gfx1100") without depending
/// on the `hipDeviceProp_t` layout: `hipGetDeviceProperties` is called into a
/// large zeroed buffer, which is then scanned for a NUL-terminated "gfx" string.
fn query_gcn_arch(lib: &HipLib, device: c_int) -> Option<String> {
    unsafe {
        let mut buf = vec![0u8; 16384];
        if (lib.hip_get_device_properties)(buf.as_mut_ptr() as *mut c_void, device) != HIP_SUCCESS {
            return None;
        }
        let mut i = 0usize;
        while i + 3 <= buf.len() {
            if buf[i] == b'g' && buf[i + 1] == b'f' && buf[i + 2] == b'x' {
                let mut end = i + 3;
                while end < buf.len() && buf[end].is_ascii_alphanumeric() {
                    end += 1;
                }
                if let Ok(s) = std::str::from_utf8(&buf[i..end]) {
                    // Must look like "gfx" followed by at least one digit.
                    if s.len() > 3 && s.as_bytes()[3].is_ascii_digit() {
                        return Some(s.to_string());
                    }
                }
            }
            i += 1;
        }
        None
    }
}

pub struct HipBackend {
    lib: HipLib,
    module: usize,   // hipModule_t
    function: usize, // hipFunction_t
    device_name: String,
    constant_buf: usize, // device pointer to the packed tail-block words
    result_buf: usize,   // device pointer to 3 u32 (lo, hi, flag)
    params_buf: usize,   // device pointer to Params
}

/// List the names of all ROCm HIP devices available on this machine.
pub fn list_devices() -> Vec<String> {
    let Some(lib) = load_hip_lib() else {
        return Vec::new();
    };
    unsafe {
        let mut count: c_int = 0;
        if (lib.hip_get_device_count)(&mut count) != HIP_SUCCESS || count <= 0 {
            return Vec::new();
        }
        (0..count)
            .map(|i| {
                let mut name = [0u8; 256];
                if (lib.hip_device_get_name)(
                    name.as_mut_ptr() as *mut c_char,
                    name.len() as c_int,
                    i,
                ) != HIP_SUCCESS
                {
                    return format!("device {i}");
                }
                String::from_utf8_lossy(&name)
                    .trim_end_matches('\0')
                    .to_string()
            })
            .collect()
    }
}

/// Probe for a ROCm HIP device (selected by `index`, defaulting to 0), compile
/// the kernel with hiprtc, and load the module. Returns `None` if the HIP
/// runtime/hiprtc libraries are absent, the index is out of range, or any step
/// fails.
pub fn detect(index: Option<usize>) -> Option<HipBackend> {
    let lib = load_hip_lib()?;

    unsafe {
        let mut count: c_int = 0;
        if (lib.hip_get_device_count)(&mut count) != HIP_SUCCESS || count <= 0 {
            return None;
        }

        let idx = index.unwrap_or(0) as c_int;
        if idx < 0 || idx >= count {
            return None;
        }

        let mut name = [0u8; 256];
        if (lib.hip_device_get_name)(
            name.as_mut_ptr() as *mut c_char,
            name.len() as c_int,
            idx,
        ) != HIP_SUCCESS
        {
            return None;
        }
        let device_name = String::from_utf8_lossy(&name)
            .trim_end_matches('\0')
            .to_string();

        let arch = query_gcn_arch(&lib, idx);
        let module = hiprtc_compile(&lib, arch.as_deref())?;

        let mut function: HipFunction = ptr::null_mut();
        if (lib.hip_module_get_function)(
            &mut function,
            module as HipModule,
            c"mine".as_ptr(),
        ) != HIP_SUCCESS
        {
            (lib.hip_module_unload)(module as HipModule);
            return None;
        }

        let mut constant_buf: *mut c_void = ptr::null_mut();
        let mut result_buf: *mut c_void = ptr::null_mut();
        let mut params_buf: *mut c_void = ptr::null_mut();
        if (lib.hip_malloc)(&mut constant_buf, CONSTANT_SIZE) != HIP_SUCCESS
            || (lib.hip_malloc)(&mut result_buf, RESULT_SIZE) != HIP_SUCCESS
            || (lib.hip_malloc)(&mut params_buf, Params::SIZE) != HIP_SUCCESS
        {
            (lib.hip_module_unload)(module as HipModule);
            return None;
        }

        Some(HipBackend {
            lib,
            module: module as usize,
            function: function as usize,
            device_name,
            constant_buf: constant_buf as usize,
            result_buf: result_buf as usize,
            params_buf: params_buf as usize,
        })
    }
}

/// Build a backend for every ROCm HIP device available on this machine.
pub fn detect_all() -> Vec<HipBackend> {
    let Some(lib) = load_hip_lib() else {
        return Vec::new();
    };
    let count = unsafe {
        let mut count: c_int = 0;
        if (lib.hip_get_device_count)(&mut count) != HIP_SUCCESS || count <= 0 {
            return Vec::new();
        }
        count
    };
    (0..count).filter_map(|i| detect(Some(i as usize))).collect()
}

/// Compile the embedded HIP source with hiprtc and load it as a module.
/// `arch` (if known) and `-O3` are passed for maximum code quality; if the
/// arch-specific compile fails, we fall back to a plain `-O3` build.
unsafe fn hiprtc_compile(lib: &HipLib, arch: Option<&str>) -> Option<HipModule> {
    unsafe {
        let src = include_str!("../shaders/sha256.hip.cpp");
        let src = CString::new(src).ok()?;
        let name = CString::new("sha256_miner").ok()?;

        let mut prog: HiprtcProgram = ptr::null_mut();
        if (lib.hiprtc_create_program)(
            &mut prog,
            src.as_ptr(),
            name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        ) != HIPRTC_SUCCESS
        {
            return None;
        }

        let o3 = CString::new("-O3").unwrap();
        let o3_ptr = o3.as_ptr() as *const c_char;
        let arch_opt = arch.map(|a| CString::new(format!("--offload-arch={a}")).unwrap());

        // Try `--offload-arch=<arch> -O3` first, then fall back to `-O3` alone
        // if the arch string is rejected.
        let compiled = match &arch_opt {
            Some(a) => {
                let ptrs = [a.as_ptr() as *const c_char, o3_ptr];
                (lib.hiprtc_compile_program)(prog, ptrs.len() as c_int, ptrs.as_ptr())
                    == HIPRTC_SUCCESS
                    || (lib.hiprtc_compile_program)(prog, 1, &o3_ptr) == HIPRTC_SUCCESS
            }
            None => (lib.hiprtc_compile_program)(prog, 1, &o3_ptr) == HIPRTC_SUCCESS,
        };

        if !compiled {
            let mut log_size: usize = 0;
            (lib.hiprtc_get_log_size)(prog, &mut log_size);
            if log_size > 1 {
                let mut log = vec![0u8; log_size];
                (lib.hiprtc_get_log)(prog, log.as_mut_ptr() as *mut c_char);
                eprintln!(
                    "[hip] hiprtc compile failed: {}",
                    String::from_utf8_lossy(&log).trim_end_matches('\0')
                );
            }
            (lib.hiprtc_destroy_program)(&mut prog);
            return None;
        }

        let mut code_size: usize = 0;
        (lib.hiprtc_get_code_size)(prog, &mut code_size);
        if code_size == 0 {
            (lib.hiprtc_destroy_program)(&mut prog);
            return None;
        }

        let mut code = vec![0u8; code_size];
        (lib.hiprtc_get_code)(prog, code.as_mut_ptr() as *mut c_char);
        (lib.hiprtc_destroy_program)(&mut prog);

        let mut module: HipModule = ptr::null_mut();
        if (lib.hip_module_load_data)(&mut module, code.as_ptr() as *const c_void) != HIP_SUCCESS {
            return None;
        }
        Some(module)
    }
}

impl HipBackend {
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

        unsafe {
            let lib = &self.lib;

            // Reset the result buffer (result[2] flag must start at 0).
            let zero = [0u32; 3];
            (lib.hip_memcpy_htod)(
                self.result_buf as *mut c_void,
                zero.as_ptr() as *const c_void,
                RESULT_SIZE,
            );

            let nonce = mine_chunks(start, count, counter, |chunk_start, c, nd| {
                let (words, num_blocks) =
                    build_tail_words(&partial_block, buffered, prefix_len, nd);
                let params = Params::new(
                    chunk_start, c, task.difficulty, buffered, nd, num_blocks, state,
                );

                (lib.hip_memcpy_htod)(
                    self.constant_buf as *mut c_void,
                    words.as_ptr() as *const c_void,
                    CONSTANT_SIZE,
                );
                (lib.hip_memcpy_htod)(
                    self.params_buf as *mut c_void,
                    &params as *const Params as *const c_void,
                    Params::SIZE,
                );

                let grid = c.div_ceil(WORKGROUP);

                let mut constant_p = self.constant_buf as *mut c_void;
                let mut result_p = self.result_buf as *mut c_void;
                let mut params_p = self.params_buf as *mut c_void;
                let kernel_params: [*mut c_void; 3] = [
                    &mut constant_p as *mut *mut c_void as *mut c_void,
                    &mut result_p as *mut *mut c_void as *mut c_void,
                    &mut params_p as *mut *mut c_void as *mut c_void,
                ];

                let err = (lib.hip_module_launch_kernel)(
                    self.function as HipFunction,
                    grid,
                    1,
                    1,
                    WORKGROUP,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    kernel_params.as_ptr() as *mut *mut c_void,
                    ptr::null_mut(),
                );
                if err != HIP_SUCCESS {
                    eprintln!("[hip] kernel launch failed: {}", lib.err_str(err));
                    return ChunkOutcome::Abort;
                }

                // Blocking readback on the default stream (implicitly ordered
                // after the kernel launch).
                let mut out = [0u32; 3];
                (lib.hip_memcpy_dtoh)(
                    out.as_mut_ptr() as *mut c_void,
                    self.result_buf as *const c_void,
                    RESULT_SIZE,
                );

                let nonce = (out[0] as u64) | ((out[1] as u64) << 32);
                if out[2] != 0 {
                    ChunkOutcome::Found(nonce)
                } else {
                    ChunkOutcome::NotFound
                }
            })?;

            Some(make_solution(task, prefix_bytes, nonce))
        }
    }
}

impl Backend for HipBackend {
    fn name(&self) -> &'static str {
        "hip"
    }

    fn description(&self) -> String {
        format!("ROCm HIP ({})", self.device_name)
    }

    fn mine(&self, task: &Task, counter: &AtomicU64) -> Option<Solution> {
        self.mine(task, counter)
    }
}

impl Drop for HipBackend {
    fn drop(&mut self) {
        unsafe {
            let lib = &self.lib;
            (lib.hip_free)(self.constant_buf as *mut c_void);
            (lib.hip_free)(self.result_buf as *mut c_void);
            (lib.hip_free)(self.params_buf as *mut c_void);
            (lib.hip_module_unload)(self.module as HipModule);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha::write_decimal;
    use crate::task::{check_difficulty, Id};

    #[test]
    fn kernel_source_sanity() {
        let src = include_str!("../shaders/sha256.hip.cpp");
        assert!(src.contains("__global__ void mine"));
        assert!(src.contains("atomicCAS"));
        assert!(src.contains("__clz"));
    }

    #[test]
    fn hip_finds_solution() {
        let Some(backend) = detect(None) else {
            eprintln!("no HIP runtime available; skipping test");
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
