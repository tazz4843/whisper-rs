#![cfg_attr(feature = "simd", feature(portable_simd))]

mod error;
mod standalone;
mod utilities;
mod whisper_ctx;
mod whisper_params;

pub use error::WhisperError;
pub use standalone::*;
pub use utilities::*;
pub use whisper_ctx::WhisperContext;
pub use whisper_params::{FullParams, SamplingStrategy};

pub type WhisperToken = std::ffi::c_int;
pub type WhisperNewSegmentCallback = whisper_rs_sys::whisper_new_segment_callback;

pub struct WhisperTokenData {
    pub id: ::std::os::raw::c_int,
    pub tid: ::std::os::raw::c_int,
    pub p: ::std::os::raw::c_float,
    pub pt: ::std::os::raw::c_float,
    pub ptsum: ::std::os::raw::c_float,
    pub t0: ::std::os::raw::c_long,
    pub t1: ::std::os::raw::c_long,
    pub vlen: ::std::os::raw::c_float,
}
