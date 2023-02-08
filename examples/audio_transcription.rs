// This example is not going to build in this folder.
// You need to copy this code into your project and add the whisper_rs dependency in your cargo.toml

use std::fs::File;
use std::io::Write;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

/// Loads a context and model, processes an audio file, and prints the resulting transcript to stdout.
fn main() {
    // Load a context and model.
    let mut ctx = WhisperContext::new("example/path/to/model/whisper.cpp/models/ggml-base.en.bin")
        .expect("failed to load model");

    // Create a params object for running the model.
    // Currently, only the Greedy sampling strategy is implemented, with BeamSearch as a WIP.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { n_past: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    params.set_n_threads(1);
    // Enable translation.
    params.set_translate(true);
    // Set the language to translate to to English.
    params.set_language("en");
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Open the audio file.
    let mut reader = hound::WavReader::open("audio.wav").expect("failed to open file");
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        ..
    } = reader.spec();

    // Convert the audio to floating point samples.
    let mut audio = whisper_rs::convert_integer_to_float_audio(
        &reader
            .samples::<i16>()
            .map(|s| s.expect("invalid sample"))
            .collect::<Vec<_>>(),
    );

    // Convert audio to 16KHz mono f32 samples, as required by the model.
    // These utilities are provided for convenience, but can be replaced with custom conversion logic.
    // SIMD variants of these functions are also available on nightly Rust (see the docs).
    if channels == 2 {
        audio = whisper_rs::convert_stereo_to_mono_audio(&audio);
    } else if channels != 1 {
        panic!(">2 channels unsupported");
    }

    if sample_rate != 16000 {
        panic!("sample rate must be 16KHz");
    }

    // Run the model.
    ctx.full(params, &audio[..]).expect("failed to run model");

    // Create a file to write the transcript to.
    let mut file = File::create("transcript.txt").expect("failed to create file");

    // Iterate through the segments of the transcript.
    let num_segments = ctx.full_n_segments();
    for i in 0..num_segments {
        // Get the transcribed text and timestamps for the current segment.
        let segment = ctx.full_get_segment_text(i).expect("failed to get segment");
        let start_timestamp = ctx.full_get_segment_t0(i);
        let end_timestamp = ctx.full_get_segment_t1(i);

        // Print the segment to stdout.
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);

        // Format the segment information as a string.
        let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);

        // Write the segment information to the file.
        file.write_all(line.as_bytes())
            .expect("failed to write to file");
    }
}
