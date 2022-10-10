use crate::error::WhisperError;
use crate::whisper_params::FullParams;
use crate::WhisperToken;
use std::ffi::{c_int, CStr, CString};

/// Safe Rust wrapper around a Whisper context.
///
/// You likely want to create this with [WhisperContext::new],
/// then run a full transcription with [WhisperContext::full].
#[derive(Debug)]
pub struct WhisperContext {
    ctx: *mut whisper_rs_sys::whisper_context,
    /// has the spectrogram been initialized in at least one way?
    spectrogram_initialized: bool,
    /// has the data been encoded?
    encode_complete: bool,
    /// has decode been called at least once?
    decode_once: bool,
}

impl WhisperContext {
    /// Create a new WhisperContext.
    ///
    /// # Arguments
    /// * path: The path to the model file.
    ///
    /// # Returns
    /// Ok(Self) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `struct whisper_context * whisper_init(const char * path_model);`
    pub fn new(path: &str) -> Result<Self, WhisperError> {
        let path_cstr = CString::new(path)?;
        let ctx = unsafe { whisper_rs_sys::whisper_init(path_cstr.as_ptr()) };
        if ctx.is_null() {
            Err(WhisperError::InitError)
        } else {
            Ok(Self {
                ctx,
                spectrogram_initialized: false,
                encode_complete: false,
                decode_once: false,
            })
        }
    }

    /// Convert raw PCM audio (floating point 32 bit) to log mel spectrogram.
    /// The resulting spectrogram is stored in the context transparently.
    ///
    /// # Arguments
    /// * pcm: The raw PCM audio.
    /// * threads: How many threads to use. Defaults to 1. Must be at least 1, returns an error otherwise.
    ///
    /// # Returns
    /// Ok(()) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_pcm_to_mel(struct whisper_context * ctx, const float * samples, int n_samples, int n_threads)`
    pub fn pcm_to_mel(&mut self, pcm: &[f32], threads: usize) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_pcm_to_mel(
                self.ctx,
                pcm.as_ptr(),
                pcm.len() as c_int,
                threads as c_int,
            )
        };
        if ret == 0 {
            self.spectrogram_initialized = true;
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// This can be used to set a custom log mel spectrogram inside the provided whisper context.
    /// Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
    ///
    /// # Note
    /// This is a low-level function.
    /// If you're a typical user, you probably don't want to use this function.
    /// See instead [WhisperContext::pcm_to_mel].
    ///
    /// # Arguments
    /// * data: The log mel spectrogram.
    ///
    /// # Returns
    /// Ok(()) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_set_mel(struct whisper_context * ctx, const float * data, int n_len, int n_mel)`
    pub fn set_mel(&mut self, data: &[f32]) -> Result<(), WhisperError> {
        let ret = unsafe {
            whisper_rs_sys::whisper_set_mel(
                self.ctx,
                data.as_ptr(),
                data.len() as c_int,
                80 as c_int,
            )
        };
        if ret == 0 {
            self.spectrogram_initialized = true;
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// Run the Whisper encoder on the log mel spectrogram stored inside the provided whisper context.
    /// Make sure to call [WhisperContext::pcm_to_mel] or [[WhisperContext::set_mel] first.
    ///
    /// # Arguments
    /// * offset: Can be used to specify the offset of the first frame in the spectrogram. Usually 0.
    /// * threads: How many threads to use. Defaults to 1. Must be at least 1, returns an error otherwise.
    ///
    /// # Returns
    /// Ok(()) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_encode(struct whisper_context * ctx, int offset, int n_threads)`
    pub fn encode(&mut self, offset: usize, threads: usize) -> Result<(), WhisperError> {
        if !self.spectrogram_initialized {
            return Err(WhisperError::SpectrogramNotInitialized);
        }
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret =
            unsafe { whisper_rs_sys::whisper_encode(self.ctx, offset as c_int, threads as c_int) };
        if ret == 0 {
            self.encode_complete = true;
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// Run the Whisper decoder to obtain the logits and probabilities for the next token.
    /// Make sure to call [WhisperContext::encode] first.
    /// tokens + n_tokens is the provided context for the decoder.
    ///
    /// # Arguments
    /// * tokens: The tokens to decode.
    /// * n_tokens: The number of tokens to decode.
    /// * n_past: The number of past tokens to use for the decoding.
    /// * n_threads: How many threads to use. Defaults to 1. Must be at least 1, returns an error otherwise.
    ///
    /// # Returns
    /// Ok(()) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_decode(struct whisper_context * ctx, const whisper_token * tokens, int n_tokens, int n_past, int n_threads)`
    pub fn decode(
        &mut self,
        tokens: &[WhisperToken],
        n_past: usize,
        threads: usize,
    ) -> Result<(), WhisperError> {
        if !self.encode_complete {
            return Err(WhisperError::EncodeNotComplete);
        }
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_decode(
                self.ctx,
                tokens.as_ptr(),
                tokens.len() as c_int,
                n_past as c_int,
                threads as c_int,
            )
        };
        if ret == 0 {
            self.decode_once = true;
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    // Token sampling functions
    /// Return the token with the highest probability.
    /// Make sure to call [WhisperContext::decode] first.
    ///
    /// # Arguments
    /// * needs_timestamp
    ///
    /// # Returns
    /// Ok(WhisperToken) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_sample_best(struct whisper_context * ctx, bool need_timestamp)`
    pub fn sample_best(&mut self, needs_timestamp: bool) -> Result<WhisperToken, WhisperError> {
        if !self.decode_once {
            return Err(WhisperError::DecodeNotComplete);
        }
        let ret = unsafe { whisper_rs_sys::whisper_sample_best(self.ctx, needs_timestamp) };
        Ok(ret)
    }

    /// Return the token with the most probable timestamp.
    /// Make sure to call [WhisperContext::decode] first.
    ///
    /// # Returns
    /// Ok(WhisperToken) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_sample_timestamp(struct whisper_context * ctx)`
    pub fn sample_timestamp(&mut self) -> Result<WhisperToken, WhisperError> {
        if !self.decode_once {
            return Err(WhisperError::DecodeNotComplete);
        }
        let ret = unsafe { whisper_rs_sys::whisper_sample_timestamp(self.ctx) };
        Ok(ret)
    }

    // model attributes
    /// Get the mel spectrogram length.
    ///
    /// # Returns
    /// Ok(c_int) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_n_len          (struct whisper_context * ctx)`
    pub fn n_len(&self) -> Result<c_int, WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_n_len(self.ctx) };
        if ret < 0 {
            Err(WhisperError::GenericError(ret))
        } else {
            Ok(ret as c_int)
        }
    }

    /// Get n_vocab.
    ///
    /// # Returns
    /// Ok(c_int) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_n_vocab        (struct whisper_context * ctx)`
    pub fn n_vocab(&self) -> Result<c_int, WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_n_vocab(self.ctx) };
        if ret < 0 {
            Err(WhisperError::GenericError(ret))
        } else {
            Ok(ret as c_int)
        }
    }

    /// Get n_text_ctx.
    ///
    /// # Returns
    /// Ok(c_int) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_n_text_ctx     (struct whisper_context * ctx)`
    pub fn n_text_ctx(&self) -> Result<c_int, WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_n_text_ctx(self.ctx) };
        if ret < 0 {
            Err(WhisperError::GenericError(ret))
        } else {
            Ok(ret as c_int)
        }
    }

    /// Does this model support multiple languages?
    ///
    /// # C++ equivalent
    /// `int whisper_is_multilingual(struct whisper_context * ctx)`
    pub fn is_multilingual(&self) -> bool {
        unsafe { whisper_rs_sys::whisper_is_multilingual(self.ctx) != 0 }
    }

    /// The probabilities for the next token.
    /// Make sure to call [WhisperContext::decode] first.
    ///
    /// # Returns
    /// Ok(*const f32) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `float * whisper_get_probs(struct whisper_context * ctx)`
    pub fn get_probs(&mut self) -> Result<*const f32, WhisperError> {
        if !self.decode_once {
            return Err(WhisperError::DecodeNotComplete);
        }
        let ret = unsafe { whisper_rs_sys::whisper_get_probs(self.ctx) };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        Ok(ret)
    }

    /// Convert a token ID to a string.
    ///
    /// # Arguments
    /// * token_id: ID of the token.
    ///
    /// # Returns
    /// Ok(String) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `const char * whisper_token_to_str(struct whisper_context * ctx, whisper_token token)`
    pub fn token_to_str(&self, token_id: WhisperToken) -> Result<String, WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_token_to_str(self.ctx, token_id) };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        let c_str = unsafe { CStr::from_ptr(ret) };
        let r_str = c_str.to_str()?;
        Ok(r_str.to_string())
    }

    // special tokens
    /// Get the ID of the eot token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_eot (struct whisper_context * ctx)`
    pub fn token_eot(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_eot(self.ctx) }
    }

    /// Get the ID of the sot token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_sot (struct whisper_context * ctx)`
    pub fn token_sot(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_sot(self.ctx) }
    }

    /// Get the ID of the prev token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_prev(struct whisper_context * ctx)`
    pub fn token_prev(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_prev(self.ctx) }
    }

    /// Get the ID of the solm token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_solm(struct whisper_context * ctx)`
    pub fn token_solm(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_solm(self.ctx) }
    }

    /// Get the ID of the not token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_not (struct whisper_context * ctx)`
    pub fn token_not(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_not(self.ctx) }
    }

    /// Get the ID of the beg token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_beg (struct whisper_context * ctx)`
    pub fn token_beg(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_beg(self.ctx) }
    }

    /// Print performance statistics to stdout.
    ///
    /// # C++ equivalent
    /// `void whisper_print_timings(struct whisper_context * ctx)`
    pub fn print_timings(&self) {
        unsafe { whisper_rs_sys::whisper_print_timings(self.ctx) }
    }

    /// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
    /// Uses the specified decoding strategy to obtain the text.
    ///
    /// This is usually the only function you need to call as an end user.
    ///
    /// # Arguments
    /// * params: [crate::FullParams] struct.
    /// * pcm: PCM audio data.
    ///
    /// # Returns
    /// Ok(c_int) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_full(struct whisper_context * ctx, struct whisper_full_params params, const float * samples, int n_samples)`
    pub fn full(&mut self, params: FullParams, data: &[f32]) -> Result<c_int, WhisperError> {
        let ret = unsafe {
            whisper_rs_sys::whisper_full(self.ctx, params.fp, data.as_ptr(), data.len() as c_int)
        };
        if ret < 0 {
            Err(WhisperError::GenericError(ret))
        } else {
            Ok(ret as c_int)
        }
    }

    /// Number of generated text segments.
    /// A segment can be a few words, a sentence, or even a paragraph.
    ///
    /// # C++ equivalent
    /// `int whisper_full_n_segments(struct whisper_context * ctx)`
    pub fn full_n_segments(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_full_n_segments(self.ctx) }
    }

    /// Get the start time of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # C++ equivalent
    /// `int64_t whisper_full_get_segment_t0(struct whisper_context * ctx, int i_segment)`
    pub fn full_get_segment_t0(&self, segment: c_int) -> i64 {
        unsafe { whisper_rs_sys::whisper_full_get_segment_t0(self.ctx, segment) }
    }

    /// Get the end time of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # C++ equivalent
    /// `int64_t whisper_full_get_segment_t1(struct whisper_context * ctx, int i_segment)`
    pub fn full_get_segment_t1(&self, segment: c_int) -> i64 {
        unsafe { whisper_rs_sys::whisper_full_get_segment_t1(self.ctx, segment) }
    }

    /// Get the text of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # Returns
    /// Ok(String) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `const char * whisper_full_get_segment_text(struct whisper_context * ctx, int i_segment)`
    pub fn full_get_segment_text(&self, segment: c_int) -> Result<String, WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_full_get_segment_text(self.ctx, segment) };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        let c_str = unsafe { CStr::from_ptr(ret) };
        let r_str = c_str.to_str()?;
        Ok(r_str.to_string())
    }
}

impl Drop for WhisperContext {
    fn drop(&mut self) {
        unsafe { whisper_rs_sys::whisper_free(self.ctx) };
    }
}
