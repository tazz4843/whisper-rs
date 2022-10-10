# whisper-rs

Rust bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp/)

## Usage
```rust
fn main() {
    // load a context and model
    let mut ctx = WhisperContext::new("path/to/model").expect("failed to load model");
    
    // create a params object
    let mut params = FullParams::new(DecodeStrategy::Greedy { n_past: 0 });

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

## Building
By default, you will experience a vague error from rustc about not being able to "find native static library \`whisper`"

To resolve this, you need to clone the original `whisper.cpp` repo and build it. Then, set `RUSTFLAGS` to point to the built library.

You also need to link against the C++ standard library, which can be done with the `-lstdc++` flag.

```shell
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make libwhisper.a
cd ..
RUSTFLAGS="-L whisper.cpp/ -Clink-args=-lstdc++" cargo build
```

## License
[Unlicense](LICENSE)

tl;dr: public domain
