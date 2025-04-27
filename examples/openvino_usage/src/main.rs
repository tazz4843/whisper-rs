#![allow(clippy::uninlined_format_args)]

use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() {
    let arg1 = std::env::args()
        .nth(1)
        .expect("first argument should be path to WAV file");
    let audio_path = Path::new(&arg1);
    if !audio_path.exists() {
        panic!("audio file doesn't exist");
    }
    let arg2 = std::env::args()
        .nth(2)
        .expect("second argument should be path to Whisper model");
    let whisper_path = Path::new(&arg2);
    if !whisper_path.exists() {
        panic!("whisper file doesn't exist")
    }

    let original_samples = examples_common::parse_wav_file(audio_path);
    let mut samples = vec![0.0f32; original_samples.len()];
    whisper_rs::convert_integer_to_float_audio(&original_samples, &mut samples)
        .expect("failed to convert samples");

    let ctx = WhisperContext::new_with_params(
        &whisper_path.to_string_lossy(),
        WhisperContextParameters::default(),
    )
    .expect("failed to open model");
    let mut state = ctx.create_state().expect("failed to create a model state");

    // Enable OpenVINO now
    // We're expecting the OpenVINO file sitting right next to the model
    state
        .init_openvino_encoder(None, "GPU", None)
        .expect("failed to enable openvino");

    let mut params = FullParams::new(SamplingStrategy::default());
    params.set_initial_prompt("experience");
    params.set_progress_callback_safe(|progress| println!("Progress callback: {}%", progress));

    let st = std::time::Instant::now();
    state
        .full(params, &samples)
        .expect("failed to convert samples");
    let et = std::time::Instant::now();

    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        let segment = state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        let start_timestamp = state
            .full_get_segment_t0(i)
            .expect("failed to get start timestamp");
        let end_timestamp = state
            .full_get_segment_t1(i)
            .expect("failed to get end timestamp");
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }
    println!("took {}ms", (et - st).as_millis());
}
