#![allow(clippy::uninlined_format_args)]

extern crate bindgen;

use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap();
    // Link C++ standard library
    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={}", cpp_stdlib);
    }
    // Link macOS Accelerate framework for matrix calculations
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        #[cfg(feature = "coreml")]
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=CoreML");
        }
        #[cfg(feature = "metal")]
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    }

    #[cfg(feature = "coreml")]
    println!("cargo:rustc-link-lib=static=whisper.coreml");
    #[cfg(feature = "opencl")]
    {
        println!("cargo:rustc-link-lib=clblast");
        println!("cargo:rustc-link-lib=OpenCL");
    }
    #[cfg(feature = "openblas")]
    {
        println!("cargo:rustc-link-lib=openblas");
    }
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        cfg_if::cfg_if! {
            if #[cfg(target_os = "windows")] {
                let cuda_path = PathBuf::from(env::var("CUDA_PATH").unwrap()).join("lib/x64");
                println!("cargo:rustc-link-search={}", cuda_path.display());
            } else {
                println!("cargo:rustc-link-lib=culibos");
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
                println!("cargo:rustc-link-search=/opt/cuda/lib64");
            }
        }
    }
    println!("cargo:rerun-if-changed=wrapper.h");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let whisper_root = out.join("whisper.cpp/");

    if !whisper_root.exists() {
        std::fs::create_dir_all(&whisper_root).unwrap();
        fs_extra::dir::copy("./whisper.cpp", &out, &Default::default()).unwrap_or_else(|e| {
            panic!(
                "Failed to copy whisper sources into {}: {}",
                whisper_root.display(),
                e
            )
        });
    }

    if env::var("WHISPER_DONT_GENERATE_BINDINGS").is_ok() {
        let _: u64 = std::fs::copy("src/bindings.rs", out.join("bindings.rs"))
            .expect("Failed to copy bindings.rs");
    } else {
        let bindings = bindgen::Builder::default()
            .header("wrapper.h")
            .clang_arg("-I./whisper.cpp")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
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
                std::fs::copy("src/bindings.rs", out.join("bindings.rs"))
                    .expect("Unable to copy bindings.rs");
            }
        }
    };

    // stop if we're on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    let mut config = Config::new(&whisper_root);

    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_ALL_WARNINGS", "OFF")
        .define("WHISPER_ALL_WARNINGS_3RD_PARTY", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .very_verbose(true)
        .pic(true);

    if cfg!(feature = "coreml") {
        config.define("WHISPER_COREML", "ON");
        config.define("WHISPER_COREML_ALLOW_FALLBACK", "1");
    }

    if cfg!(feature = "cuda") {
        config.define("WHISPER_CUBLAS", "ON");
    }

    if cfg!(feature = "openblas") {
        config.define("WHISPER_OPENBLAS", "ON");
    }

    if cfg!(feature = "opencl") {
        config.define("WHISPER_CLBLAST", "ON");
    }

    if cfg!(feature = "metal") {
        config.define("WHISPER_METAL", "ON");
    } else {
        // Metal is enabled by default, so we need to explicitly disable it
        config.define("WHISPER_METAL", "OFF");
    }

    let destination = config.build();

    if env::var("TARGET").unwrap().contains("window") {
        println!(
            "cargo:rustc-link-search={}",
            out.join("build").join("Release").display()
        );
    } else {
        println!("cargo:rustc-link-search={}", out.join("build").display());
    }
    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-lib=static=whisper");

    // for whatever reason this file is generated during build and triggers cargo complaining
    _ = std::fs::remove_file("bindings/javascript/package.json");
}

// From https://github.com/alexcrichton/cc-rs/blob/fba7feded71ee4f63cfe885673ead6d7b4f2f454/src/lib.rs#L2462
fn get_cpp_link_stdlib(target: &str) -> Option<&'static str> {
    if target.contains("msvc") {
        None
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        Some("c++")
    } else if target.contains("android") {
        Some("c++_shared")
    } else {
        Some("stdc++")
    }
}
