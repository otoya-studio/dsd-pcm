# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The halving stages (DSD128 and up) are half-band filters that skip their zero taps, and
  every filter's length is now checked against its measured response: before, Kaiser's
  length estimate left the halving stages at 114–116 dB instead of the promised 120 dB. The
  output changes by a constant fraction of a sample in timing (under 0.2 samples at 88.2 kHz);
  level, distortion and noise in the audio band are unchanged. Speed is unchanged.
- About twice as fast: each stage filters a whole block of input at a time, and the
  linear-phase filters fold their symmetric taps. The taps are now exactly symmetric, which
  changes the output in the last bits of the 64-bit float.

### Added

- `Filter` and `Decimator::with_filter`: `Filter::Audio`, flat to 20 kHz with its stopband
  from 24 kHz, removes DSD's ultrasonic noise. `Filter::Wide` stays the default.
- `Quantizer`: integer output of 8 to 32 bits with TPDF dither, reproducible by seed.
- The `dop` module: DSD over PCM 1.1 packing, unpacking with marker checks, and detection.
- A benchmark (`cargo bench`) of every DSD rate and both input layouts.

## [0.1.0]

First release.
