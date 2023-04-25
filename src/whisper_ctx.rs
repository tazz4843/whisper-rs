use crate::error::WhisperError;
use crate::whisper_params::FullParams;
use crate::whisper_state::WhisperState;
use crate::{WhisperToken, WhisperTokenData};
use std::ffi::{c_int, CStr, CString};

/// Safe Rust wrapper around a Whisper context.
///
/// You likely want to create this with [WhisperContext::new],
/// then run a full transcription with [WhisperContext::full].
#[derive(Debug)]
pub struct WhisperContext {
    ctx: *mut whisper_rs_sys::whisper_context,
}

impl WhisperContext {
    /// Create a new WhisperContext from a file.
    ///
    /// # Arguments
    /// * path: The path to the model file.
    ///
    /// # Returns
    /// Ok(Self) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `struct whisper_context * whisper_init_from_file(const char * path_model);`
    pub fn new(path: &str) -> Result<Self, WhisperError> {
        let path_cstr = CString::new(path)?;
        let ctx = unsafe { whisper_rs_sys::whisper_init_from_file_no_state(path_cstr.as_ptr()) };
        if ctx.is_null() {
            Err(WhisperError::InitError)
        } else {
            Ok(Self { ctx })
        }
    }

    /// Create a new WhisperContext from a buffer.
    ///
    /// # Arguments
    /// * buffer: The buffer containing the model.
    ///
    /// # Returns
    /// Ok(Self) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `struct whisper_context * whisper_init_from_buffer(const char * buffer, int n_bytes);`
    pub fn new_from_buffer(buffer: &[u8]) -> Result<Self, WhisperError> {
        let ctx = unsafe {
            whisper_rs_sys::whisper_init_from_buffer_no_state(buffer.as_ptr() as _, buffer.len())
        };
        if ctx.is_null() {
            Err(WhisperError::InitError)
        } else {
            Ok(Self { ctx })
        }
    }

    // we don't implement `whisper_init()` here since i have zero clue what `whisper_model_loader` does

    /// Create a new state object, ready for use.
    ///
    /// # Arguments
    /// * id: The ID of the state object. Must be unique.
    ///
    /// # Returns
    /// Ok(()) on success, Err(WhisperError) on failure.
    /// If the ID is already in use, returns Err(WhisperError::StateIdAlreadyExists).
    ///
    /// # C++ equivalent
    /// `struct whisper_state * whisper_init_state(struct whisper_context * ctx);`
    pub fn create_state(&self) -> Result<WhisperState, WhisperError> {
        let state = unsafe { whisper_rs_sys::whisper_init_state(self.ctx) };
        if state.is_null() {
            Err(WhisperError::InitError)
        } else {
            // SAFETY: this is known to be a valid pointer to a `whisper_state` struct
            Ok(WhisperState::new(state))
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
    pub fn pcm_to_mel(
        &self,
        state: &WhisperState,
        pcm: &[f32],
        threads: usize,
    ) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_pcm_to_mel_with_state(
                self.ctx,
                state.as_ptr(),
                pcm.as_ptr(),
                pcm.len() as c_int,
                threads as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::UnableToCalculateSpectrogram)
        } else if ret == 0 {
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// Convert raw PCM audio (floating point 32 bit) to log mel spectrogram.
    /// Applies a Phase Vocoder to speed up the audio x2.
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
    pub fn pcm_to_mel_phase_vocoder(
        &self,
        state: &WhisperState,
        pcm: &[f32],
        threads: usize,
    ) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_pcm_to_mel_phase_vocoder_with_state(
                self.ctx,
                state.as_ptr(),
                pcm.as_ptr(),
                pcm.len() as c_int,
                threads as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::UnableToCalculateSpectrogram)
        } else if ret == 0 {
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// This can be used to set a custom log mel spectrogram inside the provided whisper state.
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
    pub fn set_mel(&self, state: &WhisperState, data: &[f32]) -> Result<(), WhisperError> {
        let ret = unsafe {
            whisper_rs_sys::whisper_set_mel_with_state(
                self.ctx,
                state.as_ptr(),
                data.as_ptr(),
                data.len() as c_int,
                80 as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::InvalidMelBands)
        } else if ret == 0 {
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// Run the Whisper encoder on the log mel spectrogram stored inside the provided whisper context.
    /// Make sure to call [WhisperContext::pcm_to_mel] or [WhisperContext::set_mel] first.
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
    pub fn encode(
        &self,
        state: &WhisperState,
        offset: usize,
        threads: usize,
    ) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_encode_with_state(
                self.ctx,
                state.as_ptr(),
                offset as c_int,
                threads as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::UnableToCalculateEvaluation)
        } else if ret == 0 {
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
        &self,
        state: &WhisperState,
        tokens: &[WhisperToken],
        n_past: usize,
        threads: usize,
    ) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_decode_with_state(
                self.ctx,
                state.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as c_int,
                n_past as c_int,
                threads as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::UnableToCalculateEvaluation)
        } else if ret == 0 {
            Ok(())
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// Convert the provided text into tokens.
    ///
    /// # Arguments
    /// * text: The text to convert.
    ///
    /// # Returns
    /// Ok(Vec<WhisperToken>) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_tokenize(struct whisper_context * ctx, const char * text, whisper_token * tokens, int n_max_tokens);`
    pub fn tokenize(
        &self,
        text: &str,
        max_tokens: usize,
    ) -> Result<Vec<WhisperToken>, WhisperError> {
        // allocate at least max_tokens to ensure the memory is valid
        let mut tokens: Vec<WhisperToken> = Vec::with_capacity(max_tokens);
        let ret = unsafe {
            whisper_rs_sys::whisper_tokenize(
                self.ctx,
                text.as_ptr() as *const _,
                tokens.as_mut_ptr(),
                max_tokens as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::InvalidText)
        } else {
            // SAFETY: when ret != -1, we know that the length of the vector is at least ret tokens
            unsafe { tokens.set_len(ret as usize) };
            Ok(tokens)
        }
    }

    // Language functions
    /// Use mel data at offset_ms to try and auto-detect the spoken language
    /// Make sure to call pcm_to_mel() or set_mel() first
    ///
    /// # Arguments
    /// * offset_ms: The offset in milliseconds to use for the language detection.
    /// * n_threads: How many threads to use. Defaults to 1. Must be at least 1, returns an error otherwise.
    ///
    /// # Returns
    /// Ok(Vec<f32>) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_lang_auto_detect(struct whisper_context * ctx, int offset_ms, int n_threads, float * lang_probs)`
    pub fn lang_detect(
        &self,
        state: &WhisperState,
        offset_ms: usize,
        threads: usize,
    ) -> Result<Vec<f32>, WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }

        let mut lang_probs: Vec<f32> = vec![0.0; crate::standalone::get_lang_max_id() as usize + 1];
        let ret = unsafe {
            whisper_rs_sys::whisper_lang_auto_detect_with_state(
                self.ctx,
                state.as_ptr(),
                offset_ms as c_int,
                threads as c_int,
                lang_probs.as_mut_ptr(),
            )
        };
        if ret == -1 {
            Err(WhisperError::UnableToCalculateEvaluation)
        } else {
            assert_eq!(
                ret as usize,
                lang_probs.len(),
                "lang_probs length mismatch: this is a bug in whisper.cpp"
            );
            // if we're still running, double check that the length is correct, otherwise print to stderr
            // and abort, as this will cause Undefined Behavior
            // might get here due to the unwind being caught by a user-installed panic handler
            if lang_probs.len() != ret as usize {
                eprintln!(
                    "lang_probs length mismatch: this is a bug in whisper.cpp, \
                aborting to avoid Undefined Behavior"
                );
                std::process::abort();
            }
            Ok(lang_probs)
        }
    }

    // model attributes
    /// Get the mel spectrogram length.
    ///
    /// # Returns
    /// Ok(c_int) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_n_len_from_state(struct whisper_context * ctx)`
    #[inline]
    pub fn n_len(&self, state: &WhisperState) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_n_len_from_state(state.as_ptr()) })
    }

    /// Get n_vocab.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_n_vocab        (struct whisper_context * ctx)`
    #[inline]
    pub fn n_vocab(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_n_vocab(self.ctx) }
    }

    /// Get n_text_ctx.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_n_text_ctx     (struct whisper_context * ctx);`
    #[inline]
    pub fn n_text_ctx(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_n_text_ctx(self.ctx) }
    }

    /// Get n_audio_ctx.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_n_audio_ctx     (struct whisper_context * ctx);`
    #[inline]
    pub fn n_audio_ctx(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_n_audio_ctx(self.ctx) }
    }

    /// Does this model support multiple languages?
    ///
    /// # C++ equivalent
    /// `int whisper_is_multilingual(struct whisper_context * ctx)`
    #[inline]
    pub fn is_multilingual(&self) -> bool {
        unsafe { whisper_rs_sys::whisper_is_multilingual(self.ctx) != 0 }
    }

    /// Get model_n_vocab.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_vocab      (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_vocab(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_vocab(self.ctx) }
    }

    /// Get model_n_audio_ctx.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_audio_ctx    (struct whisper_context * ctx)`
    #[inline]
    pub fn model_n_audio_ctx(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_audio_ctx(self.ctx) }
    }

    /// Get model_n_audio_state.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_audio_state(struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_audio_state(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_audio_state(self.ctx) }
    }

    /// Get model_n_audio_head.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_audio_head (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_audio_head(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_audio_head(self.ctx) }
    }

    /// Get model_n_audio_layer.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_audio_layer(struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_audio_layer(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_audio_layer(self.ctx) }
    }

    /// Get model_n_text_ctx.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_text_ctx     (struct whisper_context * ctx)`
    #[inline]
    pub fn model_n_text_ctx(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_text_ctx(self.ctx) }
    }

    /// Get model_n_text_state.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_text_state (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_text_state(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_text_state(self.ctx) }
    }

    /// Get model_n_text_head.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_text_head  (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_text_head(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_text_head(self.ctx) }
    }

    /// Get model_n_text_layer.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_text_layer (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_text_layer(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_text_layer(self.ctx) }
    }

    /// Get model_n_mels.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_n_mels       (struct whisper_context * ctx);`
    #[inline]
    pub fn model_n_mels(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_n_mels(self.ctx) }
    }

    /// Get model_f16.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_f16          (struct whisper_context * ctx);`
    #[inline]
    pub fn model_f16(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_f16(self.ctx) }
    }

    /// Get model_type.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_type         (struct whisper_context * ctx);`
    #[inline]
    pub fn model_type(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_type(self.ctx) }
    }

    // logit functions
    /// Get the logits obtained from the last call to [WhisperContext::decode].
    /// The logits for the last token are stored in the last row of the matrix.
    ///
    /// Note: this function may be somewhat expensive depending on the size of the matrix returned, as it
    /// needs to be rebuilt from the raw data. Try to avoid calling it more than once if possible.
    ///
    /// # Arguments
    /// * segment: The segment to fetch data for.
    ///
    /// # Returns
    /// 2D matrix of logits. Row count is equal to n_tokens, column count is equal to n_vocab.
    ///
    /// # C++ equivalent
    /// `float * whisper_get_logits(struct whisper_context * ctx)`
    pub fn get_logits(
        &self,
        state: &WhisperState,
        segment: c_int,
    ) -> Result<Vec<Vec<f32>>, WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_get_logits_from_state(state.as_ptr()) };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        let mut logits = Vec::new();
        let n_vocab = self.n_vocab();
        let n_tokens = self.full_n_tokens(state, segment)?;
        for i in 0..n_tokens {
            let mut row = Vec::new();
            for j in 0..n_vocab {
                let idx = (i * n_vocab) + j;
                let val = unsafe { *ret.offset(idx as isize) };
                row.push(val);
            }
            logits.push(row);
        }
        Ok(logits)
    }

    // token functions
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

    /// Undocumented but exposed function in the C++ API.
    /// `const char * whisper_model_type_readable(struct whisper_context * ctx);`
    ///
    /// # Returns
    /// Ok(String) on success, Err(WhisperError) on failure.
    pub fn model_type_readable(&self) -> Result<String, WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_model_type_readable(self.ctx) };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        let c_str = unsafe { CStr::from_ptr(ret) };
        let r_str = c_str.to_str()?;
        Ok(r_str.to_string())
    }

    /// Get the ID of the eot token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_eot (struct whisper_context * ctx)`
    #[inline]
    pub fn token_eot(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_eot(self.ctx) }
    }

    /// Get the ID of the sot token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_sot (struct whisper_context * ctx)`
    #[inline]
    pub fn token_sot(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_sot(self.ctx) }
    }

    /// Get the ID of the prev token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_prev(struct whisper_context * ctx)`
    #[inline]
    pub fn token_prev(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_prev(self.ctx) }
    }

    /// Get the ID of the solm token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_solm(struct whisper_context * ctx)`
    #[inline]
    pub fn token_solm(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_solm(self.ctx) }
    }

    /// Get the ID of the not token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_not (struct whisper_context * ctx)`
    #[inline]
    pub fn token_not(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_not(self.ctx) }
    }

    /// Get the ID of the beg token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_beg (struct whisper_context * ctx)`
    #[inline]
    pub fn token_beg(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_beg(self.ctx) }
    }

    /// Get the ID of a specified language token
    ///
    /// # Arguments
    /// * lang_id: ID of the language
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_lang(struct whisper_context * ctx, int lang_id)`
    #[inline]
    pub fn token_lang(&self, lang_id: c_int) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_lang(self.ctx, lang_id) }
    }

    /// Print performance statistics to stderr.
    ///
    /// # C++ equivalent
    /// `void whisper_print_timings(struct whisper_context * ctx)`
    #[inline]
    pub fn print_timings(&self) {
        unsafe { whisper_rs_sys::whisper_print_timings(self.ctx) }
    }

    /// Reset performance statistics.
    ///
    /// # C++ equivalent
    /// `void whisper_reset_timings(struct whisper_context * ctx)`
    #[inline]
    pub fn reset_timings(&self) {
        unsafe { whisper_rs_sys::whisper_reset_timings(self.ctx) }
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
    pub fn full(
        &self,
        state: &WhisperState,
        params: FullParams,
        data: &[f32],
    ) -> Result<c_int, WhisperError> {
        let ret = unsafe {
            whisper_rs_sys::whisper_full_with_state(
                self.ctx,
                state.as_ptr(),
                params.fp,
                data.as_ptr(),
                data.len() as c_int,
            )
        };
        if ret == -1 {
            Err(WhisperError::UnableToCalculateSpectrogram)
        } else if ret == 7 {
            Err(WhisperError::FailedToEncode)
        } else if ret == 8 {
            Err(WhisperError::FailedToDecode)
        } else if ret == 0 {
            Ok(ret)
        } else {
            Err(WhisperError::GenericError(ret))
        }
    }

    /// Number of generated text segments.
    /// A segment can be a few words, a sentence, or even a paragraph.
    ///
    /// # C++ equivalent
    /// `int whisper_full_n_segments(struct whisper_context * ctx)`
    #[inline]
    pub fn full_n_segments(&self, state: &WhisperState) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_n_segments_from_state(state.as_ptr()) })
    }

    /// Language ID associated with the provided state.
    ///
    /// # C++ equivalent
    /// `int whisper_full_lang_id_from_state(struct whisper_state * state);`
    #[inline]
    pub fn full_lang_id_from_state(&self, state: &WhisperState) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_lang_id_from_state(state.as_ptr()) })
    }

    /// Get the start time of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # C++ equivalent
    /// `int64_t whisper_full_get_segment_t0(struct whisper_context * ctx, int i_segment)`
    #[inline]
    pub fn full_get_segment_t0(
        &self,
        state: &WhisperState,
        segment: c_int,
    ) -> Result<i64, WhisperError> {
        Ok(unsafe {
            whisper_rs_sys::whisper_full_get_segment_t0_from_state(state.as_ptr(), segment)
        })
    }

    /// Get the end time of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # C++ equivalent
    /// `int64_t whisper_full_get_segment_t1(struct whisper_context * ctx, int i_segment)`
    #[inline]
    pub fn full_get_segment_t1(
        &self,
        state: &WhisperState,
        segment: c_int,
    ) -> Result<i64, WhisperError> {
        Ok(unsafe {
            whisper_rs_sys::whisper_full_get_segment_t1_from_state(state.as_ptr(), segment)
        })
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
    pub fn full_get_segment_text(
        &self,
        state: &WhisperState,
        segment: c_int,
    ) -> Result<String, WhisperError> {
        let ret = unsafe {
            whisper_rs_sys::whisper_full_get_segment_text_from_state(state.as_ptr(), segment)
        };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        let c_str = unsafe { CStr::from_ptr(ret) };
        let r_str = c_str.to_str()?;
        Ok(r_str.to_string())
    }

    /// Get number of tokens in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_full_n_tokens(struct whisper_context * ctx, int i_segment)`
    #[inline]
    pub fn full_n_tokens(
        &self,
        state: &WhisperState,
        segment: c_int,
    ) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_n_tokens_from_state(state.as_ptr(), segment) })
    }

    /// Get the token text of the specified token in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// Ok(String) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `const char * whisper_full_get_token_text(struct whisper_context * ctx, int i_segment, int i_token)`
    pub fn full_get_token_text(
        &self,
        state: &WhisperState,
        segment: c_int,
        token: c_int,
    ) -> Result<String, WhisperError> {
        let ret = unsafe {
            whisper_rs_sys::whisper_full_get_token_text_from_state(
                self.ctx,
                state.as_ptr(),
                segment,
                token,
            )
        };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        let c_str = unsafe { CStr::from_ptr(ret) };
        let r_str = c_str.to_str()?;
        Ok(r_str.to_string())
    }

    /// Get the token ID of the specified token in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// [crate::WhisperToken]
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_full_get_token_id (struct whisper_context * ctx, int i_segment, int i_token)`
    pub fn full_get_token_id(
        &self,
        state: &WhisperState,
        segment: c_int,
        token: c_int,
    ) -> Result<WhisperToken, WhisperError> {
        Ok(unsafe {
            whisper_rs_sys::whisper_full_get_token_id_from_state(state.as_ptr(), segment, token)
        })
    }

    /// Get token data for the specified token in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// [crate::WhisperTokenData]
    ///
    /// # C++ equivalent
    /// `whisper_token_data whisper_full_get_token_data(struct whisper_context * ctx, int i_segment, int i_token)`
    #[inline]
    pub fn full_get_token_data(
        &self,
        state: &WhisperState,
        segment: c_int,
        token: c_int,
    ) -> Result<WhisperTokenData, WhisperError> {
        Ok(unsafe {
            whisper_rs_sys::whisper_full_get_token_data_from_state(state.as_ptr(), segment, token)
        })
    }

    /// Get the probability of the specified token in the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    /// * token: Token index.
    ///
    /// # Returns
    /// f32
    ///
    /// # C++ equivalent
    /// `float whisper_full_get_token_p(struct whisper_context * ctx, int i_segment, int i_token)`
    #[inline]
    pub fn full_get_token_prob(
        &self,
        state: &WhisperState,
        segment: c_int,
        token: c_int,
    ) -> Result<f32, WhisperError> {
        Ok(unsafe {
            whisper_rs_sys::whisper_full_get_token_p_from_state(state.as_ptr(), segment, token)
        })
    }
}

impl Drop for WhisperContext {
    #[inline]
    fn drop(&mut self) {
        unsafe { whisper_rs_sys::whisper_free(self.ctx) };
    }
}

// following implementations are safe
// see https://github.com/ggerganov/whisper.cpp/issues/32#issuecomment-1272790388
unsafe impl Send for WhisperContext {}
unsafe impl Sync for WhisperContext {}
