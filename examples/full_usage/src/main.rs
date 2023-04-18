#![allow(clippy::uninlined_format_args)]

use hound::{SampleFormat, WavReader};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

fn parse_wav_file(path: &Path) -> Vec<i16> {
    let reader = WavReader::open(path).expect("failed to read file");

    if reader.spec().channels != 1 {
        panic!("expected mono audio file");
    }
    if reader.spec().sample_format != SampleFormat::Int {
        panic!("expected integer sample format");
    }
    if reader.spec().sample_rate != 16000 {
        panic!("expected 16KHz sample rate");
    }
    if reader.spec().bits_per_sample != 16 {
        panic!("expected 16 bits per sample");
    }

    reader
        .into_samples::<i16>()
        .map(|x| x.expect("sample"))
        .collect::<Vec<_>>()
}

fn main() {
    let arg1 = std::env::args()
        .nth(1)
        .expect("first argument should be path to WAV file");
    let audio_path = Path::new(&arg1);
    if !audio_path.exists() && !audio_path.is_file() {
        panic!("expected a file");
    }
    let arg2 = std::env::args()
        .nth(2)
        .expect("second argument should be path to Whisper model");
    let whisper_path = Path::new(&arg2);
    if !whisper_path.exists() && !whisper_path.is_file() {
        panic!("expected a whisper directory")
    }

    let original_samples = parse_wav_file(audio_path);
    let samples = whisper_rs::convert_integer_to_float_audio(&original_samples);

    let ctx =
        WhisperContext::new(&whisper_path.to_string_lossy()).expect("failed to open model");
    ctx.create_key(()).expect("failed to create key");
    let params = FullParams::new(SamplingStrategy::default());

    ctx.full(&(), params, &samples)
        .expect("failed to convert samples");

    let num_segments = ctx.full_n_segments(&()).expect("failed to get number of segments");
    for i in 0..num_segments {
        let segment = ctx.full_get_segment_text(&(), i).expect("failed to get segment");
        let start_timestamp = ctx.full_get_segment_t0(&(), i).expect("failed to get start timestamp");
        let end_timestamp = ctx.full_get_segment_t1(&(), i).expect("failed to get end timestamp");
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }
}
