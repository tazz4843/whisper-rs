use std::ffi::c_int;
use std::marker::PhantomData;

pub enum DecodeStrategy {
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
    pub fn new(decode_strategy: DecodeStrategy) -> FullParams<'a> {
        let mut fp = unsafe {
            whisper_rs_sys::whisper_full_default_params(match decode_strategy {
                DecodeStrategy::Greedy { .. } => 0,
                DecodeStrategy::BeamSearch { .. } => 1,
            } as _)
        };

        match decode_strategy {
            DecodeStrategy::Greedy { n_past } => {
                // fp.__bindgen_anon_1.greedy.n_past = n_past;
            }
            DecodeStrategy::BeamSearch {
                n_past,
                beam_width,
                n_best,
            } => {
                // fp.__bindgen_anon_1.beam_search.n_past = n_past;
                // fp.__bindgen_anon_1.beam_search.beam_width = beam_width;
                // fp.__bindgen_anon_1.beam_search.n_best = n_best;
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
}

// following implementations are safe
// see https://github.com/ggerganov/whisper.cpp/issues/32#issuecomment-1272790388
// concurrent usage is prevented by &mut self on methods that modify the struct
unsafe impl<'a> Send for FullParams<'a> {}
unsafe impl<'a> Sync for FullParams<'a> {}
