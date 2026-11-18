//! Conversion speed, as a multiple of real time, for each DSD rate and both input layouts.
//!
//! ```text
//! cargo bench --bench throughput
//! ```
//!
//! A plain timing loop rather than a benchmark framework keeps the crate free of
//! dependencies; each case runs a few times and the fastest run counts.

use std::hint::black_box;
use std::time::{Duration, Instant};

use dsd_pcm::{Decimator, Filter, Quantizer, DSD64_RATE};

/// Seconds of audio per run.
const SECONDS: usize = 4;
/// Runs per case; the fastest counts.
const RUNS: usize = 15;

/// Pseudo-random bytes: the decimator does the same work for any input, and random bits
/// keep the table lookups from hitting the same cache lines.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn fastest(mut run: impl FnMut()) -> Duration {
    (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .min()
        .unwrap_or_default()
}

fn main() {
    println!("{SECONDS} s of stereo per run, fastest of {RUNS}");
    println!(
        "{:<10} {:>9} {:>12} {:>12} {:>12}",
        "rate", "output", "planar", "interleaved", "audio filter"
    );
    for multiple in [1, 2, 4, 8] {
        let dsd_rate = DSD64_RATE * multiple;
        let bytes = dsd_rate as usize / 8 * SECONDS;
        let left = noise(bytes, 1);
        let right = noise(bytes, 2);
        let interleaved: Vec<u8> = left
            .iter()
            .zip(&right)
            .flat_map(|(l, r)| [*l, *r])
            .collect();

        let mut decimator = Decimator::new(dsd_rate, 2).unwrap();
        let mut out = Vec::new();
        let planar = fastest(|| {
            out.clear();
            decimator
                .process_planar(&[&left, &right], &mut out)
                .unwrap();
            decimator.finish(&mut out);
            black_box(&out);
        });
        let interleaved = fastest(|| {
            out.clear();
            decimator
                .process_interleaved(&interleaved, &mut out)
                .unwrap();
            decimator.finish(&mut out);
            black_box(&out);
        });
        let mut audio = Decimator::new(dsd_rate, 2)
            .unwrap()
            .with_filter(Filter::Audio);
        let audio = fastest(|| {
            out.clear();
            audio.process_planar(&[&left, &right], &mut out).unwrap();
            audio.finish(&mut out);
            black_box(&out);
        });
        let realtime = |d: Duration| SECONDS as f64 / d.as_secs_f64();
        println!(
            "DSD{:<7} {:>6} Hz {:>10.0}× {:>10.0}× {:>10.0}×",
            64 * multiple,
            decimator.output_rate(),
            realtime(planar),
            realtime(interleaved),
            realtime(audio)
        );
    }

    // Quantising the output of DSD64 (88.2 kHz stereo) to 24 bits with dither.
    let samples: Vec<f64> = noise(88_200 * 2 * SECONDS, 3)
        .iter()
        .map(|&b| f64::from(b) / 255.0 - 0.5)
        .collect();
    let mut quantizer = Quantizer::new(24).unwrap();
    let mut pcm = Vec::with_capacity(samples.len());
    let quantize = fastest(|| {
        pcm.clear();
        quantizer.quantize(&samples, &mut pcm);
        black_box(&pcm);
    });
    println!(
        "24-bit TPDF quantisation of 88.2 kHz stereo: {:.0}× real time",
        SECONDS as f64 / quantize.as_secs_f64()
    );
}
