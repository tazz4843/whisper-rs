use crate::error::WhisperError;
use crate::whisper_state::WhisperState;
use crate::WhisperToken;
use std::ffi::{c_int, CStr, CString};

/// Safe Rust wrapper around a Whisper context.
///
/// You likely want to create this with [WhisperContext::new_with_params],
/// create a state with [WhisperContext::create_state],
/// then run a full transcription with [WhisperState::full].
#[derive(Debug)]
pub struct WhisperContext {
    pub(crate) ctx: *mut whisper_rs_sys::whisper_context,
}

impl WhisperContext {
    /// Create a new WhisperContext from a file, with parameters.
    ///
    /// # Arguments
    /// * path: The path to the model file.
    /// * parameters: A parameter struct containing the parameters to use.
    ///
    /// # Returns
    /// Ok(Self) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `struct whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model, struct whisper_context_params params);`
    pub fn new_with_params(
        path: &str,
        parameters: WhisperContextParameters,
    ) -> Result<Self, WhisperError> {
        let path_cstr = CString::new(path)?;
        let ctx = unsafe {
            whisper_rs_sys::whisper_init_from_file_with_params_no_state(
                path_cstr.as_ptr(),
                parameters.to_c_struct(),
            )
        };
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
    /// `struct whisper_context * whisper_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size, struct whisper_context_params params);`
    pub fn new_from_buffer_with_params(
        buffer: &[u8],
        parameters: WhisperContextParameters,
    ) -> Result<Self, WhisperError> {
        let ctx = unsafe {
            whisper_rs_sys::whisper_init_from_buffer_with_params_no_state(
                buffer.as_ptr() as _,
                buffer.len(),
                parameters.to_c_struct(),
            )
        };
        if ctx.is_null() {
            Err(WhisperError::InitError)
        } else {
            Ok(Self { ctx })
        }
    }

    /// Create a new WhisperContext from a file.
    ///
    /// # Arguments
    /// * path: The path to the model file.
    ///
    /// # Returns
    /// Ok(Self) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `struct whisper_context * whisper_init_from_file_no_state(const char * path_model)`
    #[deprecated = "Use `new_with_params` instead"]
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
    /// `struct whisper_context * whisper_init_from_buffer_no_state(void * buffer, size_t buffer_size)`
    #[deprecated = "Use `new_from_buffer_with_params` instead"]
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


    /// Convert the provided text into tokens.
    ///
    /// # Arguments
    /// * text: The text to convert.
    ///
    /// # Returns
    /// `Ok(Vec<WhisperToken>)` on success, `Err(WhisperError)` on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_tokenize(struct whisper_context * ctx, const char * text, whisper_token * tokens, int n_max_tokens);`
    pub fn tokenize(
        &self,
        text: &str,
        max_tokens: usize,
    ) -> Result<Vec<WhisperToken>, WhisperError> {
        // convert the text to a nul-terminated C string. Will raise an error if the text contains
        // any nul bytes.
        let text = CString::new(text)?;
        // allocate at least max_tokens to ensure the memory is valid
        let mut tokens: Vec<WhisperToken> = Vec::with_capacity(max_tokens);
        let ret = unsafe {
            whisper_rs_sys::whisper_tokenize(
                self.ctx,
                text.as_ptr(),
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

    /// Get model_ftype.
    ///
    /// # Returns
    /// c_int
    ///
    /// # C++ equivalent
    /// `int whisper_model_ftype          (struct whisper_context * ctx);`
    #[inline]
    pub fn model_ftype(&self) -> c_int {
        unsafe { whisper_rs_sys::whisper_model_ftype(self.ctx) }
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

    // token functions
    /// Convert a token ID to a string.
    ///
    /// # Arguments
    /// * token_id: ID of the token.
    ///
    /// # Returns
    /// Ok(&str) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `const char * whisper_token_to_str(struct whisper_context * ctx, whisper_token token)`
    pub fn token_to_str(&self, token_id: WhisperToken) -> Result<&str, WhisperError> {
        let c_str = self.token_to_cstr(token_id)?;
        let r_str = c_str.to_str()?;
        Ok(r_str)
    }

    /// Convert a token ID to a &CStr.
    ///
    /// # Arguments
    /// * token_id: ID of the token.
    ///
    /// # Returns
    /// Ok(String) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `const char * whisper_token_to_str(struct whisper_context * ctx, whisper_token token)`
    pub fn token_to_cstr(&self, token_id: WhisperToken) -> Result<&CStr, WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_token_to_str(self.ctx, token_id) };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        Ok(unsafe { CStr::from_ptr(ret) })
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

    /// Get the ID of the solm token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_solm(struct whisper_context * ctx)`
    #[inline]
    pub fn token_solm(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_solm(self.ctx) }
    }

    /// Get the ID of the prev token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_prev(struct whisper_context * ctx)`
    #[inline]
    pub fn token_prev(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_prev(self.ctx) }
    }

    /// Get the ID of the nosp token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_nosp(struct whisper_context * ctx)`
    #[inline]
    pub fn token_nosp(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_nosp(self.ctx) }
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

    // task tokens
    /// Get the ID of the translate task token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_translate ()`
    pub fn token_translate(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_translate(self.ctx) }
    }

    /// Get the ID of the transcribe task token.
    ///
    /// # C++ equivalent
    /// `whisper_token whisper_token_transcribe()`
    pub fn token_transcribe(&self) -> WhisperToken {
        unsafe { whisper_rs_sys::whisper_token_transcribe(self.ctx) }
    }

    /// Get whether the next segment is predicted as a speaker turn
    ///
    /// # Arguments
    /// * i_segment: Segment index.
    ///
    /// # Returns
    /// bool
    ///
    /// # C++ equivalent
    /// `bool whisper_full_get_segment_speaker_turn_next(struct whisper_context * ctx, int i_segment)`
    pub fn full_get_segment_speaker_turn_next(&mut self, i_segment: c_int) -> bool {
        unsafe { whisper_rs_sys::whisper_full_get_segment_speaker_turn_next(self.ctx, i_segment) }
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

pub struct WhisperContextParameters {
    /// Use GPU if available.
    ///
    /// **Warning**: Does not have an effect if OpenCL is selected as GPU backend
    /// (in that case, GPU is always enabled).
    pub use_gpu: bool,
}

#[allow(clippy::derivable_impls)] // this impl cannot be derived
impl Default for WhisperContextParameters {
    fn default() -> Self {
        Self {
            use_gpu: cfg!(feature = "_gpu"),
        }
    }
}
impl WhisperContextParameters {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn use_gpu(&mut self, use_gpu: bool) -> &mut Self {
        self.use_gpu = use_gpu;
        self
    }
    fn to_c_struct(&self) -> whisper_rs_sys::whisper_context_params {
        whisper_rs_sys::whisper_context_params {
            use_gpu: self.use_gpu,
        }
    }
}

#[cfg(test)]
#[cfg(feature = "test-with-tiny-model")]
mod test_with_tiny_model {
    use super::*;
    const MODEL_PATH: &str = "./sys/whisper.cpp/models/ggml-tiny.en.bin";

    // These tests expect that the tiny.en model has been downloaded
    // using the script `sys/whisper.cpp/models/download-ggml-model.sh tiny.en`

    #[test]
    fn test_tokenize_round_trip() {
        let ctx = WhisperContext::new(MODEL_PATH).expect("Download the ggml-tiny.en model using 'sys/whisper.cpp/models/download-ggml-model.sh tiny.en'");
        let text_in = " And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
        let tokens = ctx.tokenize(text_in, 1024).unwrap();
        let text_out = tokens
            .into_iter()
            .map(|t| ctx.token_to_str(t).unwrap())
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text_in, text_out);
    }
}
