
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

# Running on M1 OSX

To build on a M1 Mac, make sure to add the following to your project's `.cargo/config.toml`: 

```
[target.aarch64-apple-darwin]
rustflags = "-lc++ -l framework=Accelerate"
```

See https://github.com/tazz4843/whisper-rs/pull/2 for more information.
