use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

// note that running this example will not do anything, as it is just a
// demonstration of how to use the library, and actual usage requires
// more dependencies than the base library.
pub fn usage() {
    // load a context and model
    let mut ctx = WhisperContext::new("path/to/model").expect("failed to load model");

    // create a params object
    // note that currently the only implemented strategy is Greedy, BeamSearch is a WIP
    // n_past defaults to 0
    let mut params = FullParams::new(SamplingStrategy::Greedy { n_past: 0 });

    // edit things as needed
    // here we set the number of threads to use to 1
    params.set_n_threads(1);
    // we also enable translation
    params.set_translate(true);
    // and set the language to translate to to english
    params.set_language("en");
    // we also explicitly disable anything that prints to stdout
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // assume we have a buffer of audio data
    // here we'll make a fake one, integer samples, 16 bit, 16KHz, stereo
    let audio_data = vec![0_i16; 16000 * 2];

    // we must convert to 16KHz mono f32 samples for the model
    // some utilities exist for this
    // note that you don't need to use these, you can do it yourself or any other way you want
    // these are just provided for convenience
    // SIMD variants of these functions are also available, but only on nightly Rust: see the docs
    let audio_data = whisper_rs::convert_stereo_to_mono_audio(
        &whisper_rs::convert_integer_to_float_audio(&audio_data),
    );

    // now we can run the model
    ctx.full(params, &audio_data[..])
        .expect("failed to run model");

    // fetch the results
    let num_segments = ctx.full_n_segments();
    for i in 0..num_segments {
        let segment = ctx.full_get_segment_text(i).expect("failed to get segment");
        let start_timestamp = ctx.full_get_segment_t0(i);
        let end_timestamp = ctx.full_get_segment_t1(i);
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }
}

fn main() {
    println!("running this example does nothing! see the source code for usage");
}
