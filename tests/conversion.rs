//! Conversion tests: level, timing, length and the equivalence of the input layouts, checked
//! against signals with a known answer.

use std::f64::consts::PI;

use dsd_pcm::dop::{DopPacker, DopUnpacker};
use dsd_pcm::{BitOrder, Decimator, Error, Filter, DSD64_RATE};

/// A second-order sigma-delta modulation of a sine: `bits` bits at `dsd_rate`, MSB first.
fn sine(dsd_rate: u32, freq: f64, amplitude: f64, bits: usize) -> Vec<u8> {
    let (mut i1, mut i2) = (0.0f64, 0.0f64);
    let mut bytes = vec![0u8; bits / 8];
    for n in 0..bits {
        let x = amplitude * (2.0 * PI * freq * n as f64 / f64::from(dsd_rate)).sin();
        let y = if i2 >= 0.0 { 1.0 } else { -1.0 };
        i1 += x - y;
        i2 += i1 - y;
        if y > 0.0 {
            bytes[n / 8] |= 0x80 >> (n % 8);
        }
    }
    bytes
}

/// Converts whole channels in one go and finishes the stream.
fn convert<B: AsRef<[u8]>>(decimator: &mut Decimator, channels: &[B]) -> Vec<f64> {
    let mut out = Vec::new();
    decimator.process_planar(channels, &mut out).unwrap();
    decimator.finish(&mut out);
    out
}

/// Every `channels`-th sample, starting at `channel`.
fn channel(samples: &[f64], channel: usize, channels: usize) -> Vec<f64> {
    samples
        .iter()
        .skip(channel)
        .step_by(channels)
        .copied()
        .collect()
}

#[test]
fn idle_pattern_is_silence_and_dc_maps_linearly() {
    let bytes = DSD64_RATE as usize / 8 / 50; // 20 ms
    let mut decimator = Decimator::new(DSD64_RATE, 2).unwrap();
    let out = convert(&mut decimator, &[vec![0x69; bytes], vec![0xEE; bytes]]);
    assert_eq!(out.len(), 2 * bytes * 8 / 32, "one frame per 32 bits");
    // The filters settle at both ends, where the stream meets the silence around it.
    let settle = 64;
    let frames = out.len() / 2;
    for f in settle..frames - settle {
        assert!(out[2 * f].abs() < 1e-5, "0x69 is silence: {}", out[2 * f]);
        // 0xEE holds six ones and two zeros: a mean of (6 − 2) / 8 = 0.5.
        assert!(
            (out[2 * f + 1] - 0.5).abs() < 1e-5,
            "DC: {}",
            out[2 * f + 1]
        );
    }
}

#[test]
fn a_sine_keeps_its_level_and_timing() {
    let freq = 5_000.0;
    let bits = DSD64_RATE as usize / 10; // 100 ms
    let mut decimator = Decimator::new(DSD64_RATE, 1).unwrap();
    let out = convert(&mut decimator, &[sine(DSD64_RATE, freq, 0.5, bits)]);
    assert_eq!(out.len(), bits / 32);
    let rate = f64::from(decimator.output_rate());
    let (mut max_error, mut peak) = (0.0f64, 0.0f64);
    for (k, v) in out.iter().enumerate().skip(200).take(out.len() - 400) {
        let ideal = 0.5 * (2.0 * PI * freq * k as f64 / rate).sin();
        max_error = max_error.max((v - ideal).abs());
        peak = peak.max(v.abs());
    }
    assert!(
        (peak - 0.5).abs() < 0.01,
        "0 dB SACD is −6 dBFS: peak {peak}"
    );
    // One sample of misalignment would be an error of about 0.18 at 5 kHz.
    assert!(max_error < 0.01, "aligned with the input: {max_error}");
}

#[test]
fn every_rate_lands_on_the_highest_rate_below_the_limit() {
    for (dsd_rate, pcm_rate) in [
        (DSD64_RATE, 88_200),
        (2 * DSD64_RATE, 88_200),
        (4 * DSD64_RATE, 88_200),
        (8 * DSD64_RATE, 88_200),
        (3_072_000, 96_000),
        (6_144_000, 96_000),
        (12_288_000, 96_000),
    ] {
        let bits = dsd_rate as usize / 50;
        let mut decimator = Decimator::new(dsd_rate, 1).unwrap();
        assert_eq!(decimator.output_rate(), pcm_rate, "{dsd_rate} Hz");
        let out = convert(&mut decimator, &[sine(dsd_rate, 1_000.0, 0.5, bits)]);
        assert_eq!(out.len() as u64, decimator.output_frames(bits as u64));
        let peak = out[200..out.len() - 200]
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!((peak - 0.5).abs() < 0.01, "{dsd_rate} Hz: peak {peak}");
    }
}

#[test]
fn a_lower_limit_gives_a_lower_rate() {
    let decimator = Decimator::with_max_output_rate(DSD64_RATE, 2, 48_000).unwrap();
    assert_eq!(decimator.output_rate(), 44_100);
    assert_eq!(decimator.factor(), 64);
}

#[test]
fn the_output_does_not_depend_on_how_the_input_is_cut() {
    let bits = DSD64_RATE as usize / 20;
    let left = sine(DSD64_RATE, 1_000.0, 0.4, bits);
    let right = sine(DSD64_RATE, 3_000.0, 0.3, bits);
    let whole = convert(
        &mut Decimator::new(DSD64_RATE, 2).unwrap(),
        &[left.clone(), right.clone()],
    );

    let mut decimator = Decimator::new(DSD64_RATE, 2).unwrap();
    let mut pieces = Vec::new();
    let mut start = 0;
    // Odd sizes, so pieces end in the middle of output frames.
    for size in [1, 7, 333, 1021, 4093].into_iter().cycle() {
        if start == left.len() {
            break;
        }
        let end = (start + size).min(left.len());
        decimator
            .process_planar(&[&left[start..end], &right[start..end]], &mut pieces)
            .unwrap();
        start = end;
    }
    decimator.finish(&mut pieces);
    assert_eq!(whole, pieces);
}

#[test]
fn interleaved_and_planar_input_agree() {
    let bits = DSD64_RATE as usize / 50;
    let left = sine(DSD64_RATE, 2_000.0, 0.5, bits);
    let right = sine(DSD64_RATE, 500.0, 0.25, bits);
    let planar = convert(
        &mut Decimator::new(DSD64_RATE, 2).unwrap(),
        &[left.clone(), right.clone()],
    );
    let interleaved: Vec<u8> = left
        .iter()
        .zip(&right)
        .flat_map(|(l, r)| [*l, *r])
        .collect();
    let mut decimator = Decimator::new(DSD64_RATE, 2).unwrap();
    let mut out = Vec::new();
    decimator
        .process_interleaved(&interleaved, &mut out)
        .unwrap();
    decimator.finish(&mut out);
    assert_eq!(planar, out);
    assert_eq!(channel(&out, 0, 2).len(), channel(&out, 1, 2).len());
}

#[test]
fn lsb_first_input_is_read_as_reversed_bytes() {
    let bits = DSD64_RATE as usize / 50;
    let msb = sine(DSD64_RATE, 1_000.0, 0.5, bits);
    let lsb: Vec<u8> = msb.iter().map(|b| b.reverse_bits()).collect();
    let want = convert(&mut Decimator::new(DSD64_RATE, 1).unwrap(), &[msb]);
    let mut decimator = Decimator::new(DSD64_RATE, 1)
        .unwrap()
        .with_bit_order(BitOrder::LsbFirst);
    assert_eq!(want, convert(&mut decimator, &[lsb]));
}

#[test]
fn reset_starts_over_as_a_new_decimator() {
    let bits = DSD64_RATE as usize / 50;
    let signal = sine(DSD64_RATE, 1_000.0, 0.5, bits);
    let fresh = convert(&mut Decimator::new(DSD64_RATE, 1).unwrap(), &[&signal]);

    let mut decimator = Decimator::new(DSD64_RATE, 1).unwrap();
    let mut discarded = Vec::new();
    decimator
        .process_planar(&[&signal[..1000]], &mut discarded)
        .unwrap();
    decimator.reset();
    assert_eq!(convert(&mut decimator, &[signal]), fresh);
}

#[test]
fn bad_input_is_rejected_without_converting_anything() {
    assert_eq!(
        Decimator::new(DSD64_RATE, 0).unwrap_err(),
        Error::NoChannels
    );
    assert!(matches!(
        Decimator::new(2_822_401, 2),
        Err(Error::UnsupportedRate { .. })
    ));

    let mut decimator = Decimator::new(DSD64_RATE, 2).unwrap();
    let mut out = Vec::new();
    assert_eq!(
        decimator.process_interleaved(&[0x69; 3], &mut out),
        Err(Error::IncompleteFrame)
    );
    assert_eq!(
        decimator.process_planar(&[vec![0x69; 4], vec![0x69; 5]], &mut out),
        Err(Error::ChannelMismatch)
    );
    assert_eq!(
        decimator.process_planar(&[vec![0x69; 4]], &mut out),
        Err(Error::ChannelMismatch)
    );
    assert!(out.is_empty());
}

#[test]
fn the_audio_band_is_flat_and_the_noise_above_it_is_removed() {
    // A 20 kHz tone passes at full level; a tone just above the output's Nyquist frequency,
    // where DSD carries its shaped noise, must not fold back into the audio band.
    let bits = DSD64_RATE as usize / 10;
    let mut decimator = Decimator::new(DSD64_RATE, 1).unwrap();
    let pass = convert(&mut decimator, &[sine(DSD64_RATE, 20_000.0, 0.5, bits)]);
    let peak = pass[300..pass.len() - 300]
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!((peak - 0.5).abs() < 0.01, "20 kHz: peak {peak}");

    let stop = convert(&mut decimator, &[sine(DSD64_RATE, 60_000.0, 0.5, bits)]);
    let rms = (stop[300..stop.len() - 300]
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        / (stop.len() - 600) as f64)
        .sqrt();
    // The modulator's own noise sets the floor here, not the filter: a full-scale alias at
    // 28.2 kHz would show as an RMS level around 0.35.
    assert!(rms < 1e-3, "60 kHz is removed: RMS {rms}");
}

/// The RMS level of `samples` without the settling at both ends.
fn rms(samples: &[f64]) -> f64 {
    let inner = &samples[300..samples.len() - 300];
    (inner.iter().map(|v| v * v).sum::<f64>() / inner.len() as f64).sqrt()
}

#[test]
fn the_audio_filter_removes_what_the_wide_one_passes_above_20_khz() {
    let bits = DSD64_RATE as usize / 10;
    let ultrasonic = sine(DSD64_RATE, 30_000.0, 0.5, bits);
    let mut wide = Decimator::new(DSD64_RATE, 1).unwrap();
    let mut audio = Decimator::new(DSD64_RATE, 1)
        .unwrap()
        .with_filter(Filter::Audio);
    assert_eq!(wide.filter(), Filter::Wide);
    assert_eq!(audio.filter(), Filter::Audio);
    let passed = rms(&convert(&mut wide, &[&ultrasonic]));
    assert!(
        (passed - 0.5 / 2f64.sqrt()).abs() < 0.01,
        "wide passes 30 kHz: {passed}"
    );
    let removed = rms(&convert(&mut audio, &[&ultrasonic]));
    assert!(removed < 1e-3, "audio removes 30 kHz: {removed}");

    // In the audio band both keep level and timing.
    let tone = sine(DSD64_RATE, 5_000.0, 0.5, bits);
    let a = convert(&mut wide, &[&tone]);
    let b = convert(&mut audio, &[&tone]);
    assert_eq!(a.len(), b.len());
    let max = a[300..a.len() - 300]
        .iter()
        .zip(&b[300..])
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(max < 0.01, "same level and timing: {max}");
}

#[test]
fn the_audio_filter_works_at_every_output_rate() {
    for (rate, limit) in [
        (DSD64_RATE, 48_000),
        (DSD64_RATE * 4, 96_000),
        (3_072_000, 48_000),
    ] {
        let decimator = Decimator::with_max_output_rate(rate, 1, limit)
            .unwrap()
            .with_filter(Filter::Audio);
        assert!(decimator.latency() > 0, "{rate} Hz to at most {limit} Hz");
    }
}

#[test]
fn a_new_filter_keeps_the_bit_order() {
    let bits = DSD64_RATE as usize / 50;
    let msb = sine(DSD64_RATE, 1_000.0, 0.5, bits);
    let lsb: Vec<u8> = msb.iter().map(|b| b.reverse_bits()).collect();
    let mut a = Decimator::new(DSD64_RATE, 1)
        .unwrap()
        .with_filter(Filter::Audio);
    let mut b = Decimator::new(DSD64_RATE, 1)
        .unwrap()
        .with_bit_order(BitOrder::LsbFirst)
        .with_filter(Filter::Audio);
    assert_eq!(convert(&mut a, &[&msb]), convert(&mut b, &[&lsb]));
}

#[test]
fn dsd_over_pcm_carries_the_stream_unchanged() {
    let bits = DSD64_RATE as usize / 50;
    let left = sine(DSD64_RATE, 1_000.0, 0.5, bits);
    let right = sine(DSD64_RATE, 3_000.0, 0.3, bits);
    let mut words = Vec::new();
    DopPacker::new(2)
        .unwrap()
        .pack_planar(&[&left, &right], &mut words)
        .unwrap();
    assert!(dsd_pcm::dop::detect(&words, 2));
    let mut bytes = Vec::new();
    DopUnpacker::new(2)
        .unwrap()
        .unpack(&words, &mut bytes)
        .unwrap();

    let mut direct = Decimator::new(DSD64_RATE, 2).unwrap();
    let mut via_dop = Decimator::new(DSD64_RATE, 2).unwrap();
    let mut out = Vec::new();
    via_dop.process_interleaved(&bytes, &mut out).unwrap();
    via_dop.finish(&mut out);
    assert_eq!(out, convert(&mut direct, &[&left, &right]));
}
