use std::env;

fn main() {
    println!(
        "cargo:rustc-env=WHISPER_CPP_VERSION={}",
        env::var("DEP_WHISPER_WHISPER_CPP_VERSION").unwrap()
    );
}
