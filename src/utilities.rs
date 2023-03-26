#[cfg(feature = "simd")]
use std::simd::{f32x16, i16x16};

/// Convert an array of 16 bit mono audio samples to a vector of 32 bit floats.
///
/// This variant does not use SIMD instructions.
///
/// # Arguments
/// * `samples` - The array of 16 bit mono audio samples.
///
/// # Returns
/// A vector of 32 bit floats.
pub fn convert_integer_to_float_audio(samples: &[i16]) -> Vec<f32> {
    let mut floats = Vec::with_capacity(samples.len());
    for sample in samples {
        floats.push(*sample as f32 / 32768.0);
    }
    floats
}

/// Convert an array of 16 bit mono audio samples to a vector of 32 bit floats.
///
/// This variant uses SIMD instructions, and as such is only available on
/// nightly Rust.
///
/// # Arguments
/// * `samples` - The array of 16 bit mono audio samples.
///
/// # Returns
/// A vector of 32 bit floats.
#[cfg(feature = "simd")]
pub fn convert_integer_to_float_audio_simd(samples: &[i16]) -> Vec<f32> {
    let mut floats = Vec::with_capacity(samples.len());

    let div_arr = f32x16::splat(32768.0);

    let chunks = samples.chunks_exact(16);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let simd = i16x16::from_slice(chunk).cast::<f32>();
        let simd = simd / div_arr;
        floats.extend(&simd.to_array()[..]);
    }

    // Handle the remainder.
    // do this normally because it's only a few samples and the overhead of
    // converting to SIMD is not worth it.
    for sample in remainder {
        floats.push(*sample as f32 / 32768.0);
    }

    floats
}

/// Convert 32 bit floating point stereo PCM audio to 32 bit floating point mono PCM audio.
///
/// If there are an odd number of samples, the last half-sample is dropped.
/// This variant does not use SIMD instructions.
///
/// # Arguments
/// * `samples` - The array of 32 bit floating point stereo PCM audio samples.
///
/// # Returns
/// A vector of 32 bit floating point mono PCM audio samples.
pub fn convert_stereo_to_mono_audio(samples: &[f32]) -> Vec<f32> {
    samples.chunks_exact(2).map(|x| (x[0] + x[1]) / 2.0).collect()
}

/// Convert 32 bit floating point stereo PCM audio to 32 bit floating point mono PCM audio.
///
/// If there are an odd number of samples, the last half-sample is dropped.
/// This variant uses SIMD instructions, and as such is only available on
/// nightly Rust.
///
/// # Arguments
/// * `samples` - The array of 32 bit floating point stereo PCM audio samples.
///
/// # Returns
/// A vector of 32 bit floating point mono PCM audio samples.
#[cfg(feature = "simd")]
pub fn convert_stereo_to_mono_audio_simd(samples: &[f32]) -> Vec<f32> {
    let mut mono = Vec::with_capacity(samples.len() / 2);

    let div_array = f32x16::splat(2.0);

    let chunks = samples.chunks_exact(32);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let [c1, c2] = [0, 1].map(|offset| {
            let mut arr = [0.0; 16];
            std::iter::zip(&mut arr, chunk.iter().skip(offset).step_by(2).copied())
                .for_each(|(a, c)| *a = c);
            arr
        });

        let c1 = f32x16::from(c1);
        let c2 = f32x16::from(c2);
        let mono_simd = (c1 + c2) / div_array;
        mono.extend(&mono_simd.to_array()[..]);
    }

    // Handle the remainder.
    // do this normally because it's only a few samples and the overhead of
    // converting to SIMD is not worth it.
    mono.extend(convert_stereo_to_mono_audio(remainder));

    mono
}

#[cfg(feature = "simd")]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn assert_stereo_to_mono_simd() {
        // fake some sample data, of 1028 elements
        let mut samples = Vec::with_capacity(1028);
        for i in 0..1029 {
            samples.push(i as f32);
        }
        let mono_simd = convert_stereo_to_mono_audio_simd(&samples);
        let mono = convert_stereo_to_mono_audio(&samples);
        assert_eq!(mono_simd, mono);
    }
}
