use std::ffi::c_int;
use std::marker::PhantomData;

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

pub struct FullParams<'a> {
    pub(crate) fp: whisper_rs_sys::whisper_full_params,
    phantom: PhantomData<&'a str>,
}

impl<'a> FullParams<'a> {
    /// Create a new set of parameters for the decoder.
    pub fn new(sampling_strategy: SamplingStrategy) -> FullParams<'a> {
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
            phantom: PhantomData,
        }
    }

    /// Set the number of threads to use for decoding.
    ///
    /// Defaults to min(4, std::thread::hardware_concurrency()).
    pub fn set_n_threads(&mut self, n_threads: c_int) {
        self.fp.n_threads = n_threads;
    }

    /// Set the offset in milliseconds to use for decoding.
    ///
    /// Defaults to 0.
    pub fn set_offset_ms(&mut self, offset_ms: c_int) {
        self.fp.offset_ms = offset_ms;
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

    /// Set whether to print special tokens.
    ///
    /// Defaults to false.
    pub fn set_print_special_tokens(&mut self, print_special_tokens: bool) {
        self.fp.print_special_tokens = print_special_tokens;
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

    /// Set the target language.
    ///
    /// Defaults to "en".
    pub fn set_language(&mut self, language: &'a str) {
        self.fp.language = language.as_ptr() as *const _;
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
}

// following implementations are safe
// see https://github.com/ggerganov/whisper.cpp/issues/32#issuecomment-1272790388
// concurrent usage is prevented by &mut self on methods that modify the struct
unsafe impl<'a> Send for FullParams<'a> {}
unsafe impl<'a> Sync for FullParams<'a> {}
