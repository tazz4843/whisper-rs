use std::ffi::{c_char, c_int, CString};
use std::marker::PhantomData;
use whisper_rs_sys::whisper_token;

pub enum SamplingStrategy {
    Greedy {
        n_past: c_int,
    },
    /// not implemented yet, results of using this unknown
    BeamSearch {
        n_past: c_int,
        beam_width: c_int,
        n_best: c_int,
    },
}

pub struct FullParams<'a, 'b> {
    pub(crate) fp: whisper_rs_sys::whisper_full_params,
    phantom_lang: PhantomData<&'a str>,
    phantom_tokens: PhantomData<&'b [c_int]>,
}

impl<'a, 'b> FullParams<'a, 'b> {
    /// Create a new set of parameters for the decoder.
    pub fn new(sampling_strategy: SamplingStrategy) -> FullParams<'a, 'b> {
        let mut fp = unsafe {
            whisper_rs_sys::whisper_full_default_params(match sampling_strategy {
                SamplingStrategy::Greedy { .. } => {
                    whisper_rs_sys::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY
                }
                SamplingStrategy::BeamSearch { .. } => {
                    whisper_rs_sys::whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH
                }
            } as _)
        };

        match sampling_strategy {
            SamplingStrategy::Greedy { n_past } => {
                fp.greedy.n_past = n_past;
            }
            SamplingStrategy::BeamSearch {
                n_past,
                beam_width,
                n_best,
            } => {
                fp.beam_search.n_past = n_past;
                fp.beam_search.beam_width = beam_width;
                fp.beam_search.n_best = n_best;
            }
        }

        Self {
            fp,
            phantom_lang: PhantomData,
            phantom_tokens: PhantomData,
        }
    }

    /// Set the number of threads to use for decoding.
    ///
    /// Defaults to min(4, std::thread::hardware_concurrency()).
    pub fn set_n_threads(&mut self, n_threads: c_int) {
        self.fp.n_threads = n_threads;
    }

    /// Set n_max_text_ctx.
    ///
    /// Defaults to 16384.
    pub fn set_n_max_text_ctx(&mut self, n_max_text_ctx: c_int) {
        self.fp.n_max_text_ctx = n_max_text_ctx;
    }

    /// Set the start offset in milliseconds to use for decoding.
    ///
    /// Defaults to 0.
    pub fn set_offset_ms(&mut self, offset_ms: c_int) {
        self.fp.offset_ms = offset_ms;
    }

    /// Set the audio duration to process in milliseconds.
    ///
    /// Defaults to 0.
    pub fn set_duration_ms(&mut self, duration_ms: c_int) {
        self.fp.duration_ms = duration_ms;
    }

    /// Set whether to translate the output to the language specified by `language`.
    ///
    /// Defaults to false.
    pub fn set_translate(&mut self, translate: bool) {
        self.fp.translate = translate;
    }

    /// Set no_context. Usage unknown.
    ///
    /// Defaults to false.
    pub fn set_no_context(&mut self, no_context: bool) {
        self.fp.no_context = no_context;
    }

    /// Force single segment output. This may be useful for streaming.
    ///
    /// Defaults to false.
    pub fn set_single_segment(&mut self, single_segment: bool) {
        self.fp.single_segment = single_segment;
    }

    /// Set print_special. Usage unknown.
    ///
    /// Defaults to false.
    pub fn set_print_special(&mut self, print_special: bool) {
        self.fp.print_special = print_special;
    }

    /// Set whether to print progress.
    ///
    /// Defaults to true.
    pub fn set_print_progress(&mut self, print_progress: bool) {
        self.fp.print_progress = print_progress;
    }

    /// Set print_realtime. Usage unknown.
    ///
    /// Defaults to false.
    pub fn set_print_realtime(&mut self, print_realtime: bool) {
        self.fp.print_realtime = print_realtime;
    }

    /// Set whether to print timestamps.
    ///
    /// Defaults to true.
    pub fn set_print_timestamps(&mut self, print_timestamps: bool) {
        self.fp.print_timestamps = print_timestamps;
    }

    /// # EXPERIMENTAL
    ///
    /// Enable token-level timestamps.
    ///
    /// Defaults to false.
    pub fn set_token_timestamps(&mut self, token_timestamps: bool) {
        self.fp.token_timestamps = token_timestamps;
    }

    /// # EXPERIMENTAL
    ///
    /// Set timestamp token probability threshold.
    ///
    /// Defaults to 0.01.
    pub fn set_thold_pt(&mut self, thold_pt: f32) {
        self.fp.thold_pt = thold_pt;
    }

    /// # EXPERIMENTAL
    ///
    /// Set timestamp token sum probability threshold.
    ///
    /// Defaults to 0.01.
    pub fn set_thold_ptsum(&mut self, thold_ptsum: f32) {
        self.fp.thold_ptsum = thold_ptsum;
    }

    /// # EXPERIMENTAL
    ///
    /// Set maximum segment length in characters.
    ///
    /// Defaults to 0.
    pub fn set_max_len(&mut self, max_len: c_int) {
        self.fp.max_len = max_len;
    }

    /// # EXPERIMENTAL
    ///
    /// Set maximum tokens per segment. 0 means no limit.
    ///
    /// Defaults to 0.
    pub fn set_max_tokens(&mut self, max_tokens: c_int) {
        self.fp.max_tokens = max_tokens;
    }

    /// # EXPERIMENTAL
    ///
    /// Speed up audio ~2x by using phase vocoder.
    ///
    /// Defaults to false.
    pub fn set_speed_up(&mut self, speed_up: bool) {
        self.fp.speed_up = speed_up;
    }

    /// # EXPERIMENTAL
    ///
    /// Overwrite the audio context size. 0 = default.
    ///
    /// Defaults to 0.
    pub fn set_audio_ctx(&mut self, audio_ctx: c_int) {
        self.fp.audio_ctx = audio_ctx;
    }

    /// Set tokens to provide the model as initial input.
    ///
    /// These tokens are prepended to any existing text content from a previous call.
    ///
    /// Calling this more than once will overwrite the previous tokens.
    ///
    /// Defaults to an empty vector.
    pub fn set_tokens(&mut self, tokens: &'b [c_int]) {
        // turn into ptr and len
        let tokens_ptr: *const whisper_token = tokens.as_ptr();
        let tokens_len: c_int = tokens.len() as c_int;

        // set the tokens
        self.fp.prompt_tokens = tokens_ptr;
        self.fp.prompt_n_tokens = tokens_len;
    }


    /// Set the target language.
    ///
    /// Defaults to "en".
    pub fn set_language(&mut self, language: &'a str) {
        let c_lang = CString::new(language).expect("Language contains null byte");
        self.fp.language = c_lang.into_raw() as *const _;
    }

    /// Set the callback for new segments.
    ///
    /// Note that this callback has not been Rustified yet (and likely never will be, unless someone else feels the need to do so).
    /// It is still a C callback.
    ///
    /// # Safety
    /// Do not use this function unless you know what you are doing.
    /// * Be careful not to mutate the state of the whisper_context pointer returned in the callback.
    ///   This could cause undefined behavior, as this violates the thread-safety guarantees of the underlying C library.
    ///
    /// Defaults to None.
    pub unsafe fn set_new_segment_callback(
        &mut self,
        new_segment_callback: crate::WhisperNewSegmentCallback,
    ) {
        self.fp.new_segment_callback = new_segment_callback;
    }

    /// Set the user data to be passed to the new segment callback.
    ///
    /// # Safety
    /// See the safety notes for `set_new_segment_callback`.
    ///
    /// Defaults to None.
    pub unsafe fn set_new_segment_callback_user_data(&mut self, user_data: *mut std::ffi::c_void) {
        self.fp.new_segment_callback_user_data = user_data;
    }

    /// Set the callback for starting the encoder.
    ///
    /// Note that this callback has not been Rustified yet (and likely never will be, unless someone else feels the need to do so).
    /// It is still a C callback.
    ///
    /// # Safety
    /// Do not use this function unless you know what you are doing.
    /// * Be careful not to mutate the state of the whisper_context pointer returned in the callback.
    ///  This could cause undefined behavior, as this violates the thread-safety guarantees of the underlying C library.
    ///
    /// Defaults to None.
    pub unsafe fn set_start_encoder_callback(
        &mut self,
        start_encoder_callback: crate::WhisperStartEncoderCallback,
    ) {
        self.fp.encoder_begin_callback = start_encoder_callback;
    }

    /// Set the user data to be passed to the start encoder callback.
    ///
    /// # Safety
    /// See the safety notes for `set_start_encoder_callback`.
    ///
    /// Defaults to None.
    pub unsafe fn set_start_encoder_callback_user_data(&mut self, user_data: *mut std::ffi::c_void) {
        self.fp.encoder_begin_callback_user_data = user_data;
    }
}

// following implementations are safe
// see https://github.com/ggerganov/whisper.cpp/issues/32#issuecomment-1272790388
// concurrent usage is prevented by &mut self on methods that modify the struct
unsafe impl<'a, 'b> Send for FullParams<'a, 'b> {}
unsafe impl<'a, 'b> Sync for FullParams<'a, 'b> {}
