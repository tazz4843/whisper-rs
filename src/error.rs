use std::ffi::{c_int, NulError};
use std::str::Utf8Error;

#[derive(Debug, Copy, Clone)]
pub enum WhisperError {
    InitError,
    SpectrogramNotInitialized,
    EncodeNotComplete,
    DecodeNotComplete,
    InvalidThreadCount,
    InvalidUtf8 {
        error_len: Option<usize>,
        valid_up_to: usize,
    },
    NullByteInString {
        idx: usize,
    },
    NullPointer,
    GenericError(c_int),
}

impl From<Utf8Error> for WhisperError {
    fn from(e: Utf8Error) -> Self {
        Self::InvalidUtf8 {
            error_len: e.error_len(),
            valid_up_to: e.valid_up_to(),
        }
    }
}

impl From<NulError> for WhisperError {
    fn from(e: NulError) -> Self {
        Self::NullByteInString {
            idx: e.nul_position(),
        }
    }
}
