# dsd-pcm

`dsd-pcm` decimates a 1-bit DSD stream to PCM with a cascade of linear-phase FIR filters in
64-bit floating point

flat to 30 kHz, at least 120 dB of attenuation on everything
that would alias, no clipping at any legal SACD level, and output that lines up sample for
sample with the inp

```rust
use dsd_pcm::{Decimator, DSD64_RATE};

fn main() -> Result<(), dsd_pcm::Error> {
    let mut decimator = Decimator::new(DSD64_RATE, 2)?;
    assert_eq!(decimator.output_rate(), 88_200);

    let mut pcm = Vec::new();
    // Bytes interleaved by channel (DSDIFF); `process_planar` takes one slice per channel (DSF).
    decimator.process_interleaved(&[0x69; 8192], &mut pcm)?;
    // At the end of the stream, flush the filters.
    decimator.finish(&mut pcm);
    Ok(())
}
```

The output is interleaved `f64` samples, ±1.0 at full scale.

## What it does

|             |                                                                                                                |
| ----------- | -------------------------------------------------------------------------------------------------------------- |
| Input rates | DSD64 to DSD1024, 44.1 kHz and 48 kHz families; any rate a power of two above the output                       |
| Output rate | the highest DSD rate / 2ⁿ at or below a limit, 96 kHz by default (88.2 kHz from DSD64 to DSD512)               |
| Passband    | flat to 0.34 × output rate (30 kHz at 88.2 kHz), ripple below 0.001 dB; or flat to 20 kHz with `Filter::Audio` |
| Stopband    | at least 120 dB over every band that folds into the output                                                     |
| Level       | bit 1 = +1.0, bit 0 = −1.0, unity gain: 0 dB SACD (50 % modulation) is −6 dBFS                                 |
| Timing      | group delay compensated: output frame *k* is DSD bit *k* × factor; *n* bits give *n* / factor frames           |
| Input       | interleaved or planar, MSB or LSB first, in pieces of any size                                                 |
| Output      | `f64`, or integers of 8 to 32 bits with TPDF dither (`Quantizer`)                                              |
| DoP         | DSD over PCM 1.1: pack DSD into 24-bit words and unpack it again (`dop` module)                                |


The first stage divides by 8 and runs on whole bytes: its taps are grouped by 8 and each group
is a 256-entry table, so one output costs one lookup per 8 taps instead of 8 multiply-adds.
Half-band stages follow for DSD128 and up: every second tap is zero and skipped, and the length
of each is checked against its measured response, not only Kaiser's estimate. A final low-pass
divides by 4 and removes the noise DSD's noise shaping puts above the audio band. The crate
documentation explains the design in more detail.

## Filters

|               | `Filter::Wide` (default)                       | `Filter::Audio`                                                |
| ------------- | ---------------------------------------------- | -------------------------------------------------------------- |
| Flat to       | 0.34 × output rate (30 kHz at 88.2 kHz)        | 20 kHz                                                         |
| Stopband from | Nyquist frequency                              | 24 kHz (Nyquist if lower)                                      |
| For           | high-rate recordings with content above 20 kHz | removing DSD's ultrasonic noise before amplifiers and tweeters |

Both are linear-phase with at least 120 dB of attenuation and the same exact timing; set one
with `Decimator::with_filter`.

## Integer output and DoP

```rust
use dsd_pcm::dop::{DopPacker, DopUnpacker};
use dsd_pcm::{Decimator, Quantizer, DSD64_RATE};

fn main() -> Result<(), dsd_pcm::Error> {
    // DSD over PCM, for a DAC that takes DSD only that way.
    let dsd = [0x69u8; 2 * 1024];
    let mut words = Vec::new();
    DopPacker::new(2)?.pack_interleaved(&dsd, &mut words)?;
    assert!(dsd_pcm::dop::detect(&words, 2));

    // And back: DoP from a source, converted to 24-bit PCM with dither.
    let mut bytes = Vec::new();
    DopUnpacker::new(2)?.unpack(&words, &mut bytes)?;
    let mut decimator = Decimator::new(DSD64_RATE, 2)?;
    let mut pcm = Vec::new();
    decimator.process_interleaved(&bytes, &mut pcm)?;
    decimator.finish(&mut pcm);
    let mut samples = Vec::new();
    Quantizer::new(24)?.quantize(&pcm, &mut samples);
    assert_eq!(samples.len(), pcm.len());
    Ok(())
}
```

`Quantizer` adds triangular (TPDF) dither of ±1 LSB, which turns the rounding error into a
constant noise floor instead of distortion that follows the music, and clips beyond full
scale. Its noise comes from a seeded generator, so a run is reproducible.

How the input is cut does not change the output: converting a stream in one call or in
pieces of a single byte gives the same samples, bit for bit.

## Speed

Stereo conversion on one core of a 2.1 GHz Xeon (x86-64 without AVX), as a
multiple of real time:

| Input  | Output   | `Wide` | `Audio` |
| ------ | -------- | ------ | ------- |
| DSD64  | 88.2 kHz | 97×    | 44×     |
| DSD128 | 88.2 kHz | 53×    | 32×     |
| DSD256 | 88.2 kHz | 22×    | 17×     |
| DSD512 | 88.2 kHz | 12×    | 12×     |

Each channel runs through a block of input at a time, stage by stage, so the inner loops stay
short and in cache. The halving and final stages use the symmetry of their linear-phase taps
to halve the multiplications. 

`cargo bench` runs the measurement on your machine. (feel free to submit them)

## What it does not do

- It reads no files. `examples/dsf_to_wav.rs` shows a minimal DSF reader; DSDIFF and DST
  (compressed DSD) are left to the caller.
- It does not resample. To get 44.1 or 48 kHz from DSD64, set the limit with
  `Decimator::with_max_output_rate`; for any other rate, resample the output.
- It does not noise-shape. `Quantizer` dithers flat (TPDF), which is right for 24 bits; for
  16 bits a noise-shaped dither can sound quieter and is left to the caller.

## Example

```text
cargo run --release --example dsf_to_wav -- input.dsf output.wav
```


## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
