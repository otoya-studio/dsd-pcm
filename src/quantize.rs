//! Quantisation of the `f64` output to integer samples with TPDF dither.

use crate::Error;

/// Turns `f64` samples (±1.0 at full scale) into integers of a given word length, with
/// triangular (TPDF) dither of ±1 LSB.
///
/// TPDF dither makes the quantisation error independent of the signal: what is left is
/// a constant, benign noise floor (−141 dBFS RMS for 24 bits, −93 dBFS for 16) instead of
/// distortion that follows the music. Samples beyond full scale are clipped.
///
/// The dither comes from a small fixed pseudo-random generator, so the same seed and input
/// give the same output on every platform.
///
/// ```
/// use dsd_pcm::Quantizer;
///
/// let mut quantizer = Quantizer::new(24)?;
/// let mut pcm = Vec::new();
/// quantizer.quantize(&[0.5, -1.0, 2.0], &mut pcm);
/// assert!((pcm[0] - 4_194_304).abs() <= 1);
/// assert_eq!(pcm[2], 8_388_607); // clipped
/// # Ok::<(), dsd_pcm::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Quantizer {
    bits: u32,
    dither: bool,
    state: u64,
}

/// The seed [`Quantizer::new`] starts from.
const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

impl Quantizer {
    /// A quantizer to `bits` bits (8 to 32), with TPDF dither.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedBits`] if `bits` is outside 8 to 32.
    pub fn new(bits: u32) -> Result<Self, Error> {
        if !(8..=32).contains(&bits) {
            return Err(Error::UnsupportedBits { bits });
        }
        Ok(Self {
            bits,
            dither: true,
            state: DEFAULT_SEED,
        })
    }

    /// Starts the dither generator from `seed`; two quantizers with the same seed give the
    /// same output for the same input.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        // xorshift never leaves zero, so zero takes the default seed.
        self.state = if seed == 0 { DEFAULT_SEED } else { seed };
        self
    }

    /// Rounds without dither. Only for a signal that is dithered already or for tests: plain
    /// rounding turns quiet passages into distortion.
    #[must_use]
    pub fn without_dither(mut self) -> Self {
        self.dither = false;
        self
    }

    /// The word length in bits.
    #[must_use]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Quantises `samples` and appends them to `out`, right-aligned: a 24-bit sample lies in
    /// −8 388 608 ..= 8 388 607. NaN becomes 0.
    pub fn quantize(&mut self, samples: &[f64], out: &mut Vec<i32>) {
        // Exact: at most 2³¹.
        let scale = (1u64 << (self.bits - 1)) as f64;
        let max = scale - 1.0;
        out.reserve(samples.len());
        for &sample in samples {
            let dither = if self.dither {
                self.uniform() + self.uniform() - 1.0
            } else {
                0.0
            };
            let value = (sample * scale + dither).round().clamp(-scale, max);
            // In range after the clamp; NaN casts to 0.
            out.push(value as i32);
        }
    }

    /// A uniform number in [0, 1) from xorshift64*.
    fn uniform(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        // 53 bits, exact in an f64.
        bits as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_lengths_outside_8_to_32_are_refused() {
        assert_eq!(
            Quantizer::new(7).unwrap_err(),
            Error::UnsupportedBits { bits: 7 }
        );
        assert!(Quantizer::new(33).is_err());
        assert!(Quantizer::new(8).is_ok() && Quantizer::new(32).is_ok());
    }

    #[test]
    fn without_dither_it_rounds_and_clips() {
        let mut q = Quantizer::new(16).unwrap().without_dither();
        let mut out = Vec::new();
        q.quantize(
            &[0.0, 0.5, -1.0, 1.0, -2.0, f64::NAN, 1.4 / 32768.0],
            &mut out,
        );
        assert_eq!(out, [0, 16384, -32768, 32767, -32768, 0, 1]);
        let mut out = Vec::new();
        Quantizer::new(32)
            .unwrap()
            .without_dither()
            .quantize(&[1.0, -1.0, 0.25], &mut out);
        assert_eq!(out, [i32::MAX, i32::MIN, 1 << 29]);
    }

    #[test]
    fn the_dither_is_triangular_and_one_lsb_wide() {
        let mut q = Quantizer::new(16).unwrap();
        let mut out = Vec::new();
        let n = 200_000;
        q.quantize(&vec![0.0; n], &mut out);
        assert!(out.iter().all(|&v| (-1..=1).contains(&v)));
        let counts = [-1, 0, 1].map(|v| out.iter().filter(|&&x| x == v).count() as f64 / n as f64);
        // Rounding triangular noise on (−1, 1): 1/8, 3/4, 1/8.
        assert!((counts[0] - 0.125).abs() < 0.005, "{counts:?}");
        assert!((counts[1] - 0.75).abs() < 0.005, "{counts:?}");
        assert!((counts[2] - 0.125).abs() < 0.005, "{counts:?}");
    }

    #[test]
    fn dither_removes_the_bias_of_rounding() {
        // A constant a third of an LSB rounds to 0 without dither; dithered, its average is
        // kept.
        let lsb = 1.0 / 32768.0;
        let mut out = Vec::new();
        Quantizer::new(16)
            .unwrap()
            .quantize(&vec![lsb / 3.0; 300_000], &mut out);
        let mean = out.iter().map(|&v| f64::from(v)).sum::<f64>() / out.len() as f64;
        assert!((mean - 1.0 / 3.0).abs() < 0.01, "{mean}");
    }

    #[test]
    fn the_same_seed_gives_the_same_output() {
        let input: Vec<f64> = (0..1000).map(|i| f64::from(i) / 1000.0 - 0.5).collect();
        let run = |seed| {
            let mut out = Vec::new();
            Quantizer::new(24)
                .unwrap()
                .with_seed(seed)
                .quantize(&input, &mut out);
            out
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }
}
