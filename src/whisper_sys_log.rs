use log::{debug, error, info, warn};
use std::sync::Once;
use whisper_rs_sys::ggml_log_level;

unsafe extern "C" fn whisper_cpp_log_trampoline(
    level: ggml_log_level,
    text: *const std::os::raw::c_char,
    _: *mut std::os::raw::c_void, // user_data
) {
    if text.is_null() {
        error!("whisper_cpp_log_trampoline: text is nullptr");
    }

    // SAFETY: we must trust whisper.cpp that it will not pass us a string that does not satisfy
    // from_ptr's requirements.
    let log_str = unsafe { std::ffi::CStr::from_ptr(text) }
        .to_string_lossy()
        // whisper.cpp gives newlines at the end of its log messages, so we trim them
        .trim();

    match level {
        whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_DEBUG => debug!("{}", log_str),
        whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_INFO => info!("{}", log_str),
        whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_WARN => warn!("{}", log_str),
        whisper_rs_sys::ggml_log_level_GGML_LOG_LEVEL_ERROR => error!("{}", log_str),
        _ => {
            warn!(
                "whisper_cpp_log_trampoline: unknown log level {}: message: {}",
                level, log_str
            )
        }
    }
}

static LOG_TRAMPOLINE_INSTALL: Once = Once::new();

/// Shortcut utility to redirect all whisper.cpp logging to the `log` crate.
///
/// Filter for logs from the `whisper-rs` crate to see all log output from whisper.cpp.
///
/// You should only call this once (subsequent calls have no ill effect).
pub fn install_whisper_log_trampoline() {
    LOG_TRAMPOLINE_INSTALL.call_once(|| unsafe {
        whisper_rs_sys::whisper_log_set(Some(whisper_cpp_log_trampoline), std::ptr::null_mut())
    });
}
