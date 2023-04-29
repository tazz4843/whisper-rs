use crate::{FullParams, WhisperContext, WhisperError, WhisperToken, WhisperTokenData};
use std::ffi::{c_int, CStr};
use std::marker::PhantomData;

/// Rustified pointer to a Whisper state.
#[derive(Debug)]
pub struct WhisperState<'a> {
    ctx: *mut whisper_rs_sys::whisper_context,
    ptr: *mut whisper_rs_sys::whisper_state,
    _phantom: PhantomData<&'a WhisperContext>,
}

unsafe impl<'a> Send for WhisperState<'a> {}
unsafe impl<'a> Sync for WhisperState<'a> {}

impl<'a> Drop for WhisperState<'a> {
    fn drop(&mut self) {
        unsafe {
            whisper_rs_sys::whisper_free_state(self.ptr);
        }
    }
}

impl<'a> WhisperState<'a> {
    pub(crate) fn new(
        ctx: *mut whisper_rs_sys::whisper_context,
        ptr: *mut whisper_rs_sys::whisper_state,
    ) -> Self {
        Self {
            ctx,
            ptr,
            _phantom: PhantomData,
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
            whisper_rs_sys::whisper_pcm_to_mel_with_state(
                self.ctx,
                self.ptr,
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
        &mut self,
        pcm: &[f32],
        threads: usize,
    ) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_pcm_to_mel_phase_vocoder_with_state(
                self.ctx,
                self.ptr,
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
    pub fn set_mel(&mut self, data: &[f32]) -> Result<(), WhisperError> {
        let ret = unsafe {
            whisper_rs_sys::whisper_set_mel_with_state(
                self.ctx,
                self.ptr,
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

    /// Run the Whisper encoder on the log mel spectrogram stored inside the provided whisper state.
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
    pub fn encode(&mut self, offset: usize, threads: usize) -> Result<(), WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }
        let ret = unsafe {
            whisper_rs_sys::whisper_encode_with_state(
                self.ctx,
                self.ptr,
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
        &mut self,
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
                self.ptr,
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
    pub fn lang_detect(&self, offset_ms: usize, threads: usize) -> Result<Vec<f32>, WhisperError> {
        if threads < 1 {
            return Err(WhisperError::InvalidThreadCount);
        }

        let mut lang_probs: Vec<f32> = vec![0.0; crate::standalone::get_lang_max_id() as usize + 1];
        let ret = unsafe {
            whisper_rs_sys::whisper_lang_auto_detect_with_state(
                self.ctx,
                self.ptr,
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
    pub fn get_logits(&self, segment: c_int) -> Result<Vec<Vec<f32>>, WhisperError> {
        let ret = unsafe { whisper_rs_sys::whisper_get_logits_from_state(self.ptr) };
        if ret.is_null() {
            return Err(WhisperError::NullPointer);
        }
        let mut logits = Vec::new();
        let n_vocab = self.n_vocab();
        let n_tokens = self.full_n_tokens(segment)?;
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

    // model attributes
    /// Get the mel spectrogram length.
    ///
    /// # Returns
    /// Ok(c_int) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `int whisper_n_len_from_state(struct whisper_context * ctx)`
    #[inline]
    pub fn n_len(&self) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_n_len_from_state(self.ptr) })
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
            whisper_rs_sys::whisper_full_with_state(
                self.ctx,
                self.ptr,
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
    pub fn full_n_segments(&self) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_n_segments_from_state(self.ptr) })
    }

    /// Language ID associated with the provided state.
    ///
    /// # C++ equivalent
    /// `int whisper_full_lang_id_from_state(struct whisper_state * state);`
    #[inline]
    pub fn full_lang_id_from_state(&self) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_lang_id_from_state(self.ptr) })
    }

    /// Get the start time of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # C++ equivalent
    /// `int64_t whisper_full_get_segment_t0(struct whisper_context * ctx, int i_segment)`
    #[inline]
    pub fn full_get_segment_t0(&self, segment: c_int) -> Result<i64, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_get_segment_t0_from_state(self.ptr, segment) })
    }

    /// Get the end time of the specified segment.
    ///
    /// # Arguments
    /// * segment: Segment index.
    ///
    /// # C++ equivalent
    /// `int64_t whisper_full_get_segment_t1(struct whisper_context * ctx, int i_segment)`
    #[inline]
    pub fn full_get_segment_t1(&self, segment: c_int) -> Result<i64, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_get_segment_t1_from_state(self.ptr, segment) })
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
        let ret =
            unsafe { whisper_rs_sys::whisper_full_get_segment_text_from_state(self.ptr, segment) };
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
    pub fn full_n_tokens(&self, segment: c_int) -> Result<c_int, WhisperError> {
        Ok(unsafe { whisper_rs_sys::whisper_full_n_tokens_from_state(self.ptr, segment) })
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
        segment: c_int,
        token: c_int,
    ) -> Result<String, WhisperError> {
        let ret = unsafe {
            whisper_rs_sys::whisper_full_get_token_text_from_state(
                self.ctx, self.ptr, segment, token,
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
        segment: c_int,
        token: c_int,
    ) -> Result<WhisperToken, WhisperError> {
        Ok(unsafe {
            whisper_rs_sys::whisper_full_get_token_id_from_state(self.ptr, segment, token)
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
        segment: c_int,
        token: c_int,
    ) -> Result<WhisperTokenData, WhisperError> {
        Ok(unsafe {
            whisper_rs_sys::whisper_full_get_token_data_from_state(self.ptr, segment, token)
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
    pub fn full_get_token_prob(&self, segment: c_int, token: c_int) -> Result<f32, WhisperError> {
        Ok(
            unsafe {
                whisper_rs_sys::whisper_full_get_token_p_from_state(self.ptr, segment, token)
            },
        )
    }
}
