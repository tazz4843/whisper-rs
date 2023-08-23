//! Standalone functions that have no associated type.

use crate::WhisperToken;
use std::ffi::{c_int, CStr, CString};

/// Return the id of the specified language, returns -1 if not found
///
/// # Arguments
/// * lang: The language to get the id for.
///
/// # Returns
/// The ID of the language, None if not found.
///
/// # Panics
/// Panics if the language contains a null byte.
///
/// # C++ equivalent
/// `int whisper_lang_id(const char * lang)`
pub fn get_lang_id(lang: &str) -> Option<c_int> {
    let c_lang = CString::new(lang).expect("Language contains null byte");
    let ret = unsafe { whisper_rs_sys::whisper_lang_id(c_lang.as_ptr()) };
    if ret == -1 {
        None
    } else {
        Some(ret)
    }
}

/// Return the ID of the maximum language (ie the number of languages - 1)
///
/// # Returns
/// i32
///
/// # C++ equivalent
/// `int whisper_lang_max_id()`
pub fn get_lang_max_id() -> i32 {
    unsafe { whisper_rs_sys::whisper_lang_max_id() }
}

/// Get the short string of the specified language id (e.g. 2 -> "de").
///
/// # Returns
/// The short string of the language, None if not found.
///
/// # C++ equivalent
/// `const char * whisper_lang_str(int id)`
pub fn get_lang_str(id: i32) -> Option<&'static str> {
    let c_buf = unsafe { whisper_rs_sys::whisper_lang_str(id) };
    if c_buf.is_null() {
        None
    } else {
        let c_str = unsafe { CStr::from_ptr(c_buf) };
        Some(c_str.to_str().unwrap())
    }
}

// task tokens
/// Get the ID of the translate task token.
///
/// # C++ equivalent
/// `whisper_token whisper_token_translate ()`
pub fn token_translate() -> WhisperToken {
    unsafe { whisper_rs_sys::whisper_token_translate() }
}

/// Get the ID of the transcribe task token.
///
/// # C++ equivalent
/// `whisper_token whisper_token_transcribe()`
pub fn token_transcribe() -> WhisperToken {
    unsafe { whisper_rs_sys::whisper_token_transcribe() }
}

/// Print system information.
///
/// # C++ equivalent
/// `const char * whisper_print_system_info()`
pub fn print_system_info() -> &'static str {
    let c_buf = unsafe { whisper_rs_sys::whisper_print_system_info() };
    let c_str = unsafe { CStr::from_ptr(c_buf) };
    c_str.to_str().unwrap()
}

/// Programmatically exposes the information provided by `print_system_info`
///
/// # C++ equivalent
/// `int ggml_cpu_has_...`
pub struct SystemInfo {
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub blas: bool,
    pub clblast: bool,
    pub cublas: bool,
}

impl Default for SystemInfo {
    fn default() -> Self {
        unsafe {
            Self {
                avx: whisper_rs_sys::ggml_cpu_has_avx() != 0,
                avx2: whisper_rs_sys::ggml_cpu_has_avx2() != 0,
                fma: whisper_rs_sys::ggml_cpu_has_fma() != 0,
                f16c: whisper_rs_sys::ggml_cpu_has_f16c() != 0,
                blas: whisper_rs_sys::ggml_cpu_has_blas() != 0,
                clblast: whisper_rs_sys::ggml_cpu_has_clblast() != 0,
                cublas: whisper_rs_sys::ggml_cpu_has_cublas() != 0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::SystemInfo;

    #[cfg(target_os = "linux")]
    #[test]
    fn avx_enabled() {
        let cpuinfo = procfs::CpuInfo::new().expect("cpuinfo failed");
        let flags = cpuinfo.flags(0).expect("flags failed");
        let avx_enabled = flags.contains(&"avx");
        if avx_enabled {
            let info = SystemInfo::default();
            assert!(
                info.avx,
                "Whisper should be compiled with AVX support if supported by the platform"
            );
        }
    }
}
