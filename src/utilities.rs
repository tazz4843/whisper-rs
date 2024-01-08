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

/// Convert 32 bit floating point stereo PCM audio to 32 bit floating point mono PCM audio.
///
/// This variant does not use SIMD instructions.
///
/// # Arguments
/// * `samples` - The array of 32 bit floating point stereo PCM audio samples.
///
/// # Returns
/// A vector of 32 bit floating point mono PCM audio samples.
pub fn convert_stereo_to_mono_audio(samples: &[f32]) -> Result<Vec<f32>, &'static str> {
    if samples.len() & 1 != 0 {
        return Err("The stereo audio vector has an odd number of samples. \
            This means a half-sample is missing somewhere");
    }

    Ok(samples
        .chunks_exact(2)
        .map(|x| (x[0] + x[1]) / 2.0)
        .collect())
}

#[cfg(feature = "simd")]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn assert_stereo_to_mono_err() {
        // fake some sample data
        let samples = (0u16..1029).map(f32::from).collect::<Vec<f32>>();
        let mono = convert_stereo_to_mono_audio(&samples);
        assert!(mono.is_err());
    }
}
