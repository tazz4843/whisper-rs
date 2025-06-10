#![allow(clippy::uninlined_format_args)]
#![cfg_attr(test, feature(test))]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "vulkan")]
pub mod vulkan;

mod common_logging;
mod error;
mod ggml_logging_hook;
mod standalone;
mod utilities;
mod whisper_ctx;
mod whisper_ctx_wrapper;
mod whisper_grammar;
mod whisper_logging_hook;
mod whisper_params;
mod whisper_state;

pub use common_logging::GGMLLogLevel;
pub use error::WhisperError;
pub use standalone::*;
pub use utilities::*;
pub use whisper_ctx::DtwMode;
pub use whisper_ctx::DtwModelPreset;
pub use whisper_ctx::DtwParameters;
pub use whisper_ctx::WhisperContextParameters;
use whisper_ctx::WhisperInnerContext;
pub use whisper_ctx_wrapper::WhisperContext;
pub use whisper_grammar::{WhisperGrammarElement, WhisperGrammarElementType};
pub use whisper_params::{FullParams, SamplingStrategy, SegmentCallbackData};
#[cfg(feature = "raw-api")]
pub use whisper_rs_sys;
pub use whisper_state::WhisperState;

pub type WhisperSysContext = whisper_rs_sys::whisper_context;
pub type WhisperSysState = whisper_rs_sys::whisper_state;

pub type WhisperTokenData = whisper_rs_sys::whisper_token_data;
pub type WhisperToken = whisper_rs_sys::whisper_token;
pub type WhisperNewSegmentCallback = whisper_rs_sys::whisper_new_segment_callback;
pub type WhisperStartEncoderCallback = whisper_rs_sys::whisper_encoder_begin_callback;
pub type WhisperProgressCallback = whisper_rs_sys::whisper_progress_callback;
pub type WhisperLogitsFilterCallback = whisper_rs_sys::whisper_logits_filter_callback;
pub type WhisperAbortCallback = whisper_rs_sys::ggml_abort_callback;
pub type WhisperLogCallback = whisper_rs_sys::ggml_log_callback;
pub type DtwAhead = whisper_rs_sys::whisper_ahead;

/// The version of whisper.cpp that whisper-rs was linked with.
pub static WHISPER_CPP_VERSION: &str = env!("WHISPER_CPP_VERSION");

/// Redirect all whisper.cpp and GGML logs to logging hooks installed by whisper-rs.
///
/// This will stop most logs from being output to stdout/stderr and will bring them into
/// `log` or `tracing`, if the `log_backend` or `tracing_backend` features, respectively,
/// are enabled. If neither is enabled, this will essentially disable logging, as they won't
/// be output anywhere.
///
/// Note whisper.cpp and GGML do not reliably follow Rust logging conventions.
/// Use your logging crate's configuration to control how these logs will be output.
/// whisper-rs does not currently output any logs, but this may change in the future.
/// You should configure by module path and use `whisper_rs::ggml_logging_hook`,
/// and/or `whisper_rs::whisper_logging_hook`, to avoid possibly ignoring useful
/// `whisper-rs` logs in the future.
///
/// Safe to call multiple times. Only has an effect the first time.
/// (note this means installing your own logging handlers with unsafe functions after this call
/// is permanent and cannot be undone)
pub fn install_logging_hooks() {
    crate::whisper_logging_hook::install_whisper_logging_hook();
    crate::ggml_logging_hook::install_ggml_logging_hook();
}
