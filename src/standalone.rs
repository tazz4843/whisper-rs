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