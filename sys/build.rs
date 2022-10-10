extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-search=whisper.cpp");
    println!("cargo:rustc-link-lib=static=whisper");
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-I./whisper.cpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate();

    match bindings {
        Ok(b) => {
            let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
            b.write_to_file(out_path.join("bindings.rs"))
                .expect("Couldn't write bindings!");
        }
        Err(e) => {
            println!("cargo:warning=Unable to generate bindings: {}", e);
            println!("cargo:warning=Using bundled bindings.rs, which may be out of date");
            // copy src/bindings.rs to OUT_DIR
            std::fs::copy(
                "src/bindings.rs",
                env::var("OUT_DIR").unwrap() + "/bindings.rs",
            )
            .expect("Unable to copy bindings.rs");
        }
    }

    // build libwhisper.a
    env::set_current_dir("whisper.cpp").expect("Unable to change directory");
    let code = std::process::Command::new("make")
        .arg("libwhisper.a")
        .status()
        .expect("Failed to build libwhisper.a");
    if code.code() != Some(0) {
        panic!("Failed to build libwhisper.a");
    }
    // move libwhisper.a to where Cargo expects it (OUT_DIR)
    std::fs::copy(
        "libwhisper.a",
        format!("{}/libwhisper.a", env::var("OUT_DIR").unwrap()),
    )
    .expect("Failed to copy libwhisper.a");
}
