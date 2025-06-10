# OpenVINO Usage Example

Run `cargo build --release` in this directory,
then `./target/release/openvino_usage ../examples_common/2830-3890-0043.wav /path/to/ggml-model.bin`

There should be an OpenVINO file associated with the model next to it,
otherwise you will get an error at runtime.

## Getting your paws on OpenVINO data

Unfortunately there's no downloads of OpenVINO state. The only way to get it is generating it.

Example for most Linux distros (run this from the current directory):

```bash
cd ../..

# We need to pull in whisper.cpp.
# This should've already been done when you cloned the repo, but let's be sure.
git submodule update --init --recursive

cd sys/whisper.cpp/models/

# Generate a new venv and install the required things.
# This might take a bit, grab a drink.
# (yes this installs CUDA even if you don't have a Nvidia GPU, enjoy your 6GB venv setup)
python3.12 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-openvino.txt

# This is the key line. Change base as necessary to the name of the model you want.
python3 convert-whisper-to-openvino.py --model base
```

Do note a line that states
`assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"`
is not fatal.
The output file will still be generated normally.

See upstream's README for more info: https://github.com/ggerganov/whisper.cpp/#openvino-support
