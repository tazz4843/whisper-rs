use std::ffi::{c_int, NulError};
use std::str::Utf8Error;

/// Whisper tends to output errors to stderr, so if an error occurs, check stderr.
#[derive(Debug, Copy, Clone)]
pub enum WhisperError {
    /// Failed to create a new context.
    InitError,
    /// User didn't initialize spectrogram
    SpectrogramNotInitialized,
    /// Encode was not called.
    EncodeNotComplete,
    /// Decode was not called.
    DecodeNotComplete,
    /// Failed to calculate the spectrogram for some reason.
    UnableToCalculateSpectrogram,
    /// Failed to evaluate model.
    UnableToCalculateEvaluation,
    /// Failed to run the encoder
    FailedToEncode,
    /// Failed to run the decoder
    FailedToDecode,
    /// Invalid number of mel bands.
    InvalidMelBands,
    /// Invalid thread count
    InvalidThreadCount,
    /// Invalid UTF-8 detected in a string from Whisper.
    InvalidUtf8 {
        error_len: Option<usize>,
        valid_up_to: usize,
    },
    /// A null byte was detected in a user-provided string.
    NullByteInString { idx: usize },
    /// Whisper returned a null pointer.
    NullPointer,
    /// Generic whisper error. Varies depending on the function.
    GenericError(c_int),
    /// Whisper failed to convert the provided text into tokens.
    InvalidText,
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
