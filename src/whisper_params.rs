use std::ffi::{c_float, c_int, CString};
use std::marker::PhantomData;
use whisper_rs_sys::whisper_token;

pub enum SamplingStrategy {
    Greedy {
        best_of: c_int,
    },
    BeamSearch {
        beam_size: c_int,
        // not implemented in whisper.cpp as of this writing (v1.2.0)
        patience: c_float,
    },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Greedy { best_of: 1 }
    }
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
            SamplingStrategy::Greedy { best_of } => {
                fp.greedy.best_of = best_of;
            }
            SamplingStrategy::BeamSearch {
                beam_size,
                patience,
            } => {
                fp.beam_search.beam_size = beam_size;
                fp.beam_search.patience = patience;
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

    /// Max tokens to use from past text as prompt for the decoder
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

    /// Do not use past transcription (if any) as initial prompt for the decoder.
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

    /// Print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
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

    /// Print results from within whisper.cpp.
    /// Try to use the callback methods instead: [set_new_segment_callback](FullParams::set_new_segment_callback),
    /// [set_new_segment_callback_user_data](FullParams::set_new_segment_callback_user_data).
    ///
    /// Defaults to false.
    pub fn set_print_realtime(&mut self, print_realtime: bool) {
        self.fp.print_realtime = print_realtime;
    }

    /// Print timestamps for each text segment when printing realtime. Only has an effect if
    /// [set_print_realtime](FullParams::set_print_realtime) is set to true.
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
    /// Should the timestamps be split on words instead of characters?
    ///
    /// Defaults to false.
    pub fn set_split_on_word(&mut self, split_on_word: bool) {
        self.fp.split_on_word = split_on_word;
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
    /// Note that this can significantly reduce the accuracy of the transcription.
    ///
    /// Defaults to false.
    pub fn set_speed_up(&mut self, speed_up: bool) {
        self.fp.speed_up = speed_up;
    }

    /// # EXPERIMENTAL
    ///
    /// Overwrite the audio context size. 0 = default.
    /// As with [set_speed_up](FullParams::set_speed_up), this can significantly reduce the accuracy of the transcription.
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
    /// For auto-detection, set this to either "auto" or None.
    ///
    /// Defaults to "en".
    pub fn set_language(&mut self, language: Option<&'a str>) {
        self.fp.language = match language {
            Some(language) => CString::new(language)
                .expect("Language contains null byte")
                .into_raw() as *const _,
            None => std::ptr::null(),
        };
    }

    /// Set suppress_blank. See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
    /// for more information.
    ///
    /// Defaults to true.
    pub fn set_suppress_blank(&mut self, suppress_blank: bool) {
        self.fp.suppress_blank = suppress_blank;
    }

    /// Set suppress_non_speech_tokens. See https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253
    /// for more information.
    ///
    /// Defaults to false.
    pub fn set_suppress_non_speech_tokens(&mut self, suppress_non_speech_tokens: bool) {
        self.fp.suppress_non_speech_tokens = suppress_non_speech_tokens;
    }

    /// Set initial decoding temperature. See https://ai.stackexchange.com/a/32478 for more information.
    ///
    /// Defaults to 0.0.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.fp.temperature = temperature;
    }

    /// Set max_initial_ts. See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
    /// for more information.
    ///
    /// Defaults to 1.0.
    pub fn set_max_initial_ts(&mut self, max_initial_ts: f32) {
        self.fp.max_initial_ts = max_initial_ts;
    }

    /// Set length_penalty. See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267
    /// for more information.
    ///
    /// Defaults to -1.0.
    pub fn set_length_penalty(&mut self, length_penalty: f32) {
        self.fp.length_penalty = length_penalty;
    }

    /// Set temperature_inc. See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278
    /// for more information.
    ///
    /// Defaults to 0.2.
    pub fn set_temperature_inc(&mut self, temperature_inc: f32) {
        self.fp.temperature_inc = temperature_inc;
    }

    /// Set entropy_thold. Similar to OpenAI's compression_ratio_threshold.
    /// See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278 for more information.
    ///
    /// Defaults to 2.4.
    pub fn set_entropy_thold(&mut self, entropy_thold: f32) {
        self.fp.entropy_thold = entropy_thold;
    }

    /// Set logprob_thold. See https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278
    /// for more information.
    ///
    /// Defaults to -1.0.
    pub fn set_logprob_thold(&mut self, logprob_thold: f32) {
        self.fp.logprob_thold = logprob_thold;
    }

    /// Set no_speech_thold. Currently (as of v1.3.0) not implemented.
    ///
    /// Defaults to 0.6.
    pub fn set_no_speech_thold(&mut self, no_speech_thold: f32) {
        self.fp.no_speech_thold = no_speech_thold;
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

    /// Set the callback for progress updates.
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
    pub unsafe fn set_progress_callback(
        &mut self,
        progress_callback: crate::WhisperProgressCallback,
    ) {
        self.fp.progress_callback = progress_callback;
    }

    /// Set the user data to be passed to the progress callback.
    ///
    /// # Safety
    /// See the safety notes for `set_progress_callback`.
    ///
    /// Defaults to None.
    pub unsafe fn set_progress_callback_user_data(&mut self, user_data: *mut std::ffi::c_void) {
        self.fp.progress_callback_user_data = user_data;
    }

    /// Set the callback that is called each time before the encoder begins.
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
    pub unsafe fn set_start_encoder_callback_user_data(
        &mut self,
        user_data: *mut std::ffi::c_void,
    ) {
        self.fp.encoder_begin_callback_user_data = user_data;
    }

    /// Set the callback that is called by each decoder to filter obtained logits.
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
    pub unsafe fn set_filter_logits_callback(
        &mut self,
        logits_filter_callback: crate::WhisperLogitsFilterCallback,
    ) {
        self.fp.logits_filter_callback = logits_filter_callback;
    }

    /// Set the user data to be passed to the logits filter callback.
    ///
    /// # Safety
    /// See the safety notes for `set_filter_logits_callback`.
    ///
    /// Defaults to None.
    pub unsafe fn set_filter_logits_callback_user_data(
        &mut self,
        user_data: *mut std::ffi::c_void,
    ) {
        self.fp.logits_filter_callback_user_data = user_data;
    }
}

// following implementations are safe
// see https://github.com/ggerganov/whisper.cpp/issues/32#issuecomment-1272790388
// concurrent usage is prevented by &mut self on methods that modify the struct
unsafe impl<'a, 'b> Send for FullParams<'a, 'b> {}
unsafe impl<'a, 'b> Sync for FullParams<'a, 'b> {}
