#![allow(clippy::uninlined_format_args)]

use std::path::Path;
use std::{thread, time};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{LocalRb, Rb, SharedRb};

use std::io::Write;

use std::cell::RefCell;
use webrtc_vad::{SampleRate, Vad};

const LATENCY_MS: f32 = 1000.0;

thread_local! {
    static VAD: RefCell<Vad> = RefCell::new(Vad::new_with_rate(SampleRate::Rate48kHz));
}

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.min(max).max(min)
}

fn make_audio_louder(audio_samples: &[f32], gain: f32) -> Vec<f32> {
    audio_samples
        .iter()
        .map(|sample| {
            let louder_sample = sample * gain;
            clamp(louder_sample, -1.0, 1.0)
        })
        .collect()
}

pub fn run_example() -> Result<(), anyhow::Error> {
    let host = cpal::default_host();

    // Default devices.
    let input_device = host
        .default_input_device()
        .expect("failed to get default input device");
    let output_device = host
        .default_output_device()
        .expect("failed to get default output device");
    println!("Using default input device: \"{}\"", input_device.name()?);
    println!("Using default output device: \"{}\"", output_device.name()?);

    // We'll try and use the same configuration between streams to keep it simple.
    let config: cpal::StreamConfig = input_device.default_input_config()?.into();

    // Create a delay in case the input and output devices aren't synced.
    let latency_frames = (LATENCY_MS / 1_000.0) * config.sample_rate.0 as f32;
    let latency_samples = latency_frames as usize * config.channels as usize * 5;
    println!("{}", config.sample_rate.0);

    // The buffer to share samples
    let ring = SharedRb::<f32, _>::new(latency_samples);
    let (mut producer, mut consumer) = ring.split();

    // Vad setup
    let vad_size = (10 * config.sample_rate.0 / 1_000) as usize;
    let mut vad_ring = SharedRb::<i16, _>::new(vad_size);

    // Fill the samples with 0s
    for _ in 0..vad_size {
        vad_ring.push(0).unwrap();
    }

    let mut vad_ring_full = vec![0_i16; vad_size];
    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut output_fell_behind = false;
        let samples = whisper_rs::convert_stereo_to_mono_audio(data).unwrap();
        let samples = make_audio_louder(&samples, 4.0);

        // TODO: JPB: Make temp array up here so that copy_from_slice only needs to be called once

        for sample in samples {
            // Add sample to vad ring buffer
            let sample_i16 = (sample * i16::MAX as f32) as i16;
            vad_ring.push_overwrite(sample_i16);

            // Get all the sample from the vad ring buffer in order
            let (head, tail) = vad_ring.as_slices();
            vad_ring_full[0..head.len()].copy_from_slice(head);
            vad_ring_full[head.len()..].copy_from_slice(tail);

            // Check if person is talking
            let mut talking = false;
            VAD.with(|vad| {
                talking = vad
                    .borrow_mut()
                    .is_voice_segment(&vad_ring_full[..])
                    .unwrap();
            });
            //println!("talking: {talking}");

            // TODO: JPB: talking is a problem due to spacing...

            // Write to main ring buffer if talking
            //if talking {
            if producer.push(sample).is_err() {
                output_fell_behind = true;
            }
            //}
        }

        if output_fell_behind {
            eprintln!("output stream fell behind: try increasing latency");
        }
    };

    //let output_data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
    //    let mut input_fell_behind = None;
    //    for sample in data {
    //        *sample = match consumer.pop() {
    //            Ok(s) => s,
    //            Err(err) => {
    //                input_fell_behind = Some(err);
    //                0.0
    //            }
    //        };
    //    }
    //    if let Some(err) = input_fell_behind {
    //        eprintln!(
    //            "input stream fell behind: {:?}: try increasing latency",
    //            err
    //        );
    //    }
    //};

    // Build streams.
    println!(
        "Attempting to build the streams with f32 samples and `{:?}`.",
        config
    );
    println!("Setup input stream");
    let input_stream = input_device.build_input_stream(&config, input_data_fn, err_fn, None)?;
    //println!("Setup output stream");
    //let output_stream = output_device.build_output_stream(&config, output_data_fn, err_fn, None)?;
    println!("Successfully built streams.");

    // WHISPER SETUP

    let arg1 = std::env::args()
        .nth(1)
        .expect("First argument should be path to Whisper model");
    let whisper_path = Path::new(&arg1);
    if !whisper_path.exists() && !whisper_path.is_file() {
        panic!("expected a whisper directory")
    }

    let mut ctx =
        WhisperContext::new(&whisper_path.to_string_lossy()).expect("failed to open model");

    // START EVERYTHING

    // Play the streams.
    println!(
        "Starting the input and output streams with `{}` milliseconds of latency.",
        LATENCY_MS
    );
    thread::sleep(time::Duration::from_millis(1000));
    input_stream.play()?;
    //output_stream.play()?;

    let mut final_ring = LocalRb::new(latency_samples);
    let mut samples = vec![0_f32; latency_samples as usize];
    let mut iterations = 0;
    loop {
        // Only run the model once a second
        thread::sleep(time::Duration::from_millis(1000));

        // Go to a new line every five seconds
        iterations += 1;
        if iterations > 5 {
            iterations = 0;
            final_ring.clear();
            samples.iter_mut().map(|x| *x = 0.0).count();
            println!("");
        }

        // Make the model params
        let mut params = FullParams::new(SamplingStrategy::default());
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(true);
        params.set_language(Some("en"));
        params.set_suppress_blank(true);
        params.set_no_speech_thold(0.3);
        params.set_no_context(true);

        // Get the new samples
        final_ring.push_iter_overwrite(consumer.pop_iter());
        let (head, tail) = final_ring.as_slices();
        samples[0..head.len()].copy_from_slice(head);
        samples[head.len()..head.len() + tail.len()].copy_from_slice(tail);

        // [TESTING] Only use the samples from the last second
        //let mut samples = vec![0_f32; final_ring.len() as usize];
        //final_ring.pop_slice(&mut samples[..]);

        // Run the model
        ctx.full(params, &samples)
            .expect("failed to convert samples");

        // Output the results
        let num_segments = ctx.full_n_segments();
        for i in 0..num_segments {
            let segment = ctx.full_get_segment_text(i).expect("failed to get segment");
            let words = segment
                .replace("[BLANK_AUDIO]", "")
                .replace("[ Silence ]", "")
                .trim_end()
                .to_owned();
            print!("\x1B[2K\r{words}");
            std::io::stdout().flush().unwrap();
            //println!("{words} ");
        }
    }
}

fn err_fn(err: cpal::StreamError) {
    eprintln!("an error occurred on stream: {}", err);
}

fn main() -> Result<(), anyhow::Error> {
    run_example()
}
