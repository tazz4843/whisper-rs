# whisper-rs

Rust bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp/)

## Usage
```rust
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy};

fn main() {
    // load a context and model
    let mut ctx = WhisperContext::new("path/to/model").expect("failed to load model");
    
    // create a params object
    let mut params = FullParams::new(SamplingStrategy::Greedy { n_past: 0 });

    // assume we have a buffer of audio data
    // here we'll make a fake one, floating point samples, 32 bit, 16KHz, mono
    let audio_data = vec![0_f32; 16000 * 2];

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
```

See [examples/basic_use.rs](examples/basic_use.rs) for more details.

Lower level bindings are exposed if needed, but the above should be enough for most use cases.
See the docs: https://docs.rs/whisper-rs/ for more details.

## Troubleshooting

* I get an error about a lot of undefined symbols at compile time!
  * These symbols might be part of the C++ standard library.
    * Try linking against it with the `-Clink-args=-lstdc++` compiler flag: 
    * `RUSTFLAGS="-Clink-args=-lstdc++" cargo build`
* Windows/macOS/Android aren't working!
  * I don't have a way to test these platforms, so I can't really help you.
    * If you can get it working, please open a PR!
* I get a panic during binding generation build!
  * You can attempt to fix it yourself, or you can set the `WHISPER_DONT_GENERATE_BINDINGS` environment variable.
    This skips attempting to build the bindings whatsoever and copies the existing ones. They may be out of date,
    but it's better than nothing.
    * `WHISPER_DONT_GENERATE_BINDINGS=1 cargo build`
  * If you can fix the issue, please open a PR!
* M1 build info:
  * See [this issue](https://github.com/tazz4843/whisper-rs/pull/2) for more info.

## License
[Unlicense](LICENSE)

tl;dr: public domain
