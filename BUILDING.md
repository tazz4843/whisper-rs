# Running on Arch Linux
`sudo pacman -Syy llvm clang cmake`
`cargo build`

# Running on Windows using MSYS2

The following are instructions for building whisper-rs on Windows using the msys2 set of compilers. 

1. install msys2/mingw by following [https://code.visualstudio.com/docs/cpp/config-mingw](`https://code.visualstudio.com/docs/cpp/config-mingw`)
   1. Install g++ and make within msys2 ucrt64
      - `pacman -S --needed base-devel mingw-w64-x86_64-toolchain`
   2. Add the msys2 ucrt64 bin folder to path `C:\msys64\ucrt64\bin`
2. Install make by running `pacman -S make` in msys2 ucrt66
3. Set rust to use msys2: by running `rustup toolchain install stable-x86_64-pc-windows-gnu` in Windows Powershell/Cmd
4. Add `.cargo/config.toml` file in the project with the following contents: 
```
[target.x86_64-pc-windows-gnu]
linker = "C:\\msys64\\ucrt64\\bin\\gcc.exe"
ar = "C:\\msys64\\ucrt64\\bin\\ar.exe"
```
5. Run `cargo run`  in Windows Powershell/Cmd

# Running on Windows using Microsoft Visual Studio C++

It has been reported that it is also possible to build whisper-rs using Visual Studio C++.

Make sure you have installed and in the path:

- Visual Studio C++
- cmake
- LLVM(clang)

### Instructions (for builds with `cuda` enabled)
1. Download [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows)
2. Download [Visual Studio with Desktop C++ and Clang enabled](https://visualstudio.microsoft.com/de/downloads/) (see clang link below for installer walkthrough)
3. Download [CLANG](https://www.wikihow.com/Install-Clang-on-Windows)
4. Download [CMAKE](https://cmake.org/download/)
5. Run `where.exe clang`, then `setx LIBCLANG_PATH "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin"` or something like that
6. Restart your shell!!!
7. Cargo build

# Running on M1 OSX

To build on a M1 Mac, make sure to add the following to your project's `.cargo/config.toml`: 

```
[target.aarch64-apple-darwin]
rustflags = "-lc++ -l framework=Accelerate"
```

See https://github.com/tazz4843/whisper-rs/pull/2 for more information.

You also need to have CMake installed. You can obtain this using homebrew:

```
brew install cmake
```

CMake can also be installed from https://cmake.org/download/ but `cmake` binary needs to be in your PATH.

# OpenVINO support

## Development Tools
OpenVINO support requires the OpenVINO Development Tools to be installed. You can find
instructions for installing the OpenVINO Development Tools here:
https://docs.openvino.ai/2023.0/openvino_docs_install_guides_install_dev_tools.html#for-c-developers

On Arch Linux, you can install the OpenVINO Development Tools with the following command:
```
paru -S openvino
```
This build may take a significant amount of time, but can save massive headaches later on.

## Building
First, the `openvino` feature must be enabled in your Cargo.toml.

Next, you must set the `OpenVINO_DIR` environment variable to the path where CMake can find
`OpenVINOConfig.cmake`.
This is usually in the `cmake` directory of the OpenVINO installation.

If you used the AUR package to install OpenVINO, the location of this file is `/opt/intel/openvino/runtime/cmake`.

```
export OpenVINO_DIR=/opt/intel/openvino/runtime/cmake
```

Finally, you can build whisper-rs as normal.

## Tested platforms
- Arch Linux

If you have successfully built whisper-rs with OpenVINO on another platform, please open a PR to document it here!