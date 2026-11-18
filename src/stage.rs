//! The filter stages of the decimation cascade.

/// Table index that stands for silence: it contributes 0.0 to every sum. Used for the history
/// before the first byte and for flushing the filters at the end of a stream.
pub(crate) const SILENCE: u16 = 256;

/// Entries per table: the 256 byte values and [`SILENCE`], rounded up to a power of two.
const TABLE_LEN: usize = 512;

/// The first stage's FIR, evaluated a byte at a time through lookup tables.
///
/// The filter runs at the DSD rate and keeps every 8th output. Its taps are cut into groups of
/// 8; for each group a table holds the group's sum for all 256 byte values. One output is then
/// one table lookup per group instead of one multiply-add per bit.
#[derive(Debug, Clone)]
pub(crate) struct ByteTables {
    /// `tables[j][byte]`: the contribution of the `j`-th byte of a window that runs oldest
    /// first, so the newest byte meets the last table. Index [`SILENCE`] is zero, and so is everything above it: the tables are padded to
    /// [`TABLE_LEN`] so that a masked index needs no bounds check.
    tables: Vec<[f64; TABLE_LEN]>,
}

impl ByteTables {
    /// Tables for `taps`, whose length is a multiple of 8. Bytes are MSB first: the most
    /// significant bit is the earliest sample.
    pub(crate) fn new(taps: &[f64]) -> Self {
        debug_assert_eq!(taps.len() % 8, 0, "stage 1 length must be a multiple of 8");
        // Groups from the newest byte back, then reversed into window order.
        let mut tables: Vec<[f64; TABLE_LEN]> = taps
            .chunks_exact(8)
            .map(|group| {
                let mut table = [0.0; TABLE_LEN];
                for (byte, entry) in table.iter_mut().enumerate().take(256) {
                    // Bit `i` of the newest byte (counted from the MSB, the earliest bit) meets
                    // tap `7 - i`: the latest bit is the one with the smallest delay.
                    *entry = (0..8)
                        .map(|i| {
                            let level = if (byte >> (7 - i)) & 1 == 1 {
                                1.0
                            } else {
                                -1.0
                            };
                            group[7 - i] * level
                        })
                        .sum();
                }
                table
            })
            .collect();
        tables.reverse();
        Self { tables }
    }

    /// Number of bytes the filter spans.
    pub(crate) fn len(&self) -> usize {
        self.tables.len()
    }
}

/// The first stage of one channel: the byte history and the table lookups.
#[derive(Debug, Clone)]
pub(crate) struct ByteFilter {
    /// Table indices, oldest first: the `len − 1` bytes before the block, then the block.
    /// Starts with silence.
    bytes: Vec<u16>,
}

impl ByteFilter {
    /// A filter whose history is `len` bytes of silence.
    pub(crate) fn new(len: usize) -> Self {
        Self {
            bytes: vec![SILENCE; len.saturating_sub(1)],
        }
    }

    /// Filters a block of table indices (bytes or [`SILENCE`]), appending one output per
    /// index.
    pub(crate) fn process(
        &mut self,
        indices: impl Iterator<Item = u16>,
        tables: &ByteTables,
        out: &mut Vec<f64>,
    ) {
        self.bytes.extend(indices);
        let len = tables.len();
        out.extend(
            self.bytes
                .windows(len)
                .map(|window| lookup(window, &tables.tables)),
        );
        self.bytes.drain(..self.bytes.len() + 1 - len);
    }
}

/// The sum of one table entry per byte of `window`.
#[inline]
fn lookup(window: &[u16], tables: &[[f64; TABLE_LEN]]) -> f64 {
    // The mask keeps every index inside a table, so the compiler drops the bounds check;
    // indices in the window are at most SILENCE and pass it unchanged.
    let index = |b: u16| usize::from(b) & (TABLE_LEN - 1);
    // Two accumulators break the dependency chain of the lookups.
    let mut acc = [0.0f64; 2];
    for (k, (table, &b)) in tables.iter().zip(window).enumerate() {
        acc[k & 1] += table[index(b)];
    }
    acc[0] + acc[1]
}

/// A decimating FIR stage in 64-bit float. It computes only the outputs it keeps.
///
/// The filters are linear-phase, so their taps are symmetric: the stage adds the two samples
/// that meet the same tap before multiplying, which halves the multiplications. Symmetry also
/// means the window can be read oldest first, so the input is kept in plain order.
#[derive(Debug, Clone)]
pub(crate) struct FirDecimator {
    /// The first half of the taps; the second half mirrors it. For a half-band filter only
    /// every second tap, the others being zero.
    half: Vec<f64>,
    /// 2 for a half-band filter, whose taps at even distances from the centre are zero and
    /// skipped; 1 otherwise.
    stride: usize,
    /// The centre tap of a filter with an odd number of taps.
    centre: Option<f64>,
    len: usize,
    factor: usize,
    /// Input, oldest first: the samples the next output still needs and those after them.
    /// Starts with `len − 1` samples of silence.
    buffer: Vec<f64>,
    /// Index in `buffer` of the newest sample of the next output's window.
    next: usize,
}

impl FirDecimator {
    /// A stage that filters with `taps`, which must be symmetric, and keeps every
    /// `factor`-th output.
    pub(crate) fn new(taps: &[f64], factor: usize) -> Self {
        debug_assert!(
            taps.iter().zip(taps.iter().rev()).all(|(a, b)| a == b),
            "the taps of a linear-phase filter are symmetric"
        );
        let len = taps.len().max(1);
        let factor = factor.max(1);
        let half = &taps[..taps.len() / 2];
        // With 4k + 3 taps the centre is at an odd index, so its even distances fall on the
        // odd indices of the first half.
        let halfband = len % 4 == 3 && half.iter().skip(1).step_by(2).all(|&tap| tap == 0.0);
        let stride = if halfband { 2 } else { 1 };
        Self {
            half: half.iter().step_by(stride).copied().collect(),
            stride,
            centre: (taps.len() % 2 == 1).then(|| taps[taps.len() / 2]),
            len,
            factor,
            buffer: vec![0.0; len - 1],
            // The first output comes with the `factor`-th input.
            next: len - 1 + factor - 1,
        }
    }

    /// Number of taps.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Filters a block of input samples, appending one output per `factor` inputs.
    pub(crate) fn process(&mut self, input: &[f64], out: &mut Vec<f64>) {
        self.buffer.extend_from_slice(input);
        let h = self.len / 2;
        while self.next < self.buffer.len() {
            let window = &self.buffer[self.next + 1 - self.len..=self.next];
            let (front, back) = (&window[..h], &window[self.len - h..]);
            let folded = if self.stride == 2 {
                folded_dot_halfband(front, back, &self.half)
            } else {
                folded_dot(front, back, &self.half)
            };
            out.push(folded + self.centre.map_or(0.0, |c| c * window[h]));
            self.next += self.factor;
        }
        // Keep only what later windows reach back to.
        let used = (self.next + 1 - self.len).min(self.buffer.len());
        self.buffer.drain(..used);
        self.next -= used;
    }
}

/// `Σ taps[i] · (front[i] + back[n − 1 − i])`: a symmetric filter's dot product, with the
/// samples that share a tap added first.
///
/// Four independent accumulators let the compiler vectorise the loop; a single running sum
/// would fix the order of the additions and keep it scalar.
fn folded_dot(front: &[f64], back: &[f64], taps: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let mut fronts = front.chunks_exact(4);
    let mut backs = back.rchunks_exact(4);
    let mut tap_chunks = taps.chunks_exact(4);
    for ((f, b), t) in (&mut fronts).zip(&mut backs).zip(&mut tap_chunks) {
        for i in 0..4 {
            acc[i] += t[i] * (f[i] + b[3 - i]);
        }
    }
    let tail: f64 = fronts
        .remainder()
        .iter()
        .zip(backs.remainder().iter().rev())
        .zip(tap_chunks.remainder())
        .map(|((f, b), t)| t * (f + b))
        .sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// [`folded_dot`] for a half-band filter: `taps` holds every second tap of the first half,
/// which meet `front[2i]` and `back[h − 1 − 2i]`. The half length `h` is odd, so the last tap
/// meets the last front and the first back sample.
fn folded_dot_halfband(front: &[f64], back: &[f64], taps: &[f64]) -> f64 {
    let mut acc = 0.0;
    let fronts = front.chunks_exact(2);
    let backs = back.rchunks_exact(2);
    let (last_front, last_back) = (fronts.remainder(), backs.remainder());
    let mut tap_iter = taps.iter();
    for ((f, b), t) in fronts.zip(backs).zip(&mut tap_iter) {
        acc += t * (f[0] + b[1]);
    }
    let tail = match (tap_iter.next(), last_front, last_back) {
        (Some(t), [f], [b]) => t * (f + b),
        _ => 0.0,
    };
    acc + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_agree_with_bitwise_evaluation() {
        let taps: Vec<f64> = (0..16).map(|i| f64::from(i) * 0.25 - 1.0).collect();
        let tables = ByteTables::new(&taps);
        let mut filter = ByteFilter::new(tables.len());
        let bytes = [0b1011_0010u8, 0x5A, 0xFF];
        let mut bits = Vec::new();
        for &byte in &bytes {
            let mut out = Vec::new();
            filter.process(std::iter::once(u16::from(byte)), &tables, &mut out);
            let got = out[0];
            for i in (0..8).rev() {
                bits.push(if (byte >> i) & 1 == 1 { 1.0 } else { -1.0 });
            }
            // Direct convolution: tap 0 meets the latest bit; before the stream, silence.
            let want: f64 = taps
                .iter()
                .enumerate()
                .map(|(k, tap)| bits.len().checked_sub(1 + k).map_or(0.0, |i| tap * bits[i]))
                .sum();
            assert!(
                (got - want).abs() < 1e-12,
                "byte {byte:#04x}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn decimator_keeps_every_factor_th_output() {
        let mut stage = FirDecimator::new(&[0.5, 0.5], 2);
        let mut out = Vec::new();
        stage.process(&[1.0, 3.0], &mut out);
        stage.process(&[5.0], &mut out);
        stage.process(&[7.0], &mut out);
        assert_eq!(out, [2.0, 6.0]);
    }

    #[test]
    fn a_half_band_stage_skips_its_zeros_and_gives_the_plain_result() {
        let taps = [0.1, 0.0, -0.3, 0.0, 0.9, 0.5, 0.9, 0.0, -0.3, 0.0, 0.1];
        let stage = FirDecimator::new(&taps, 2);
        assert_eq!(
            (stage.stride, stage.half.as_slice()),
            (2, &[0.1, -0.3, 0.9][..])
        );
        let samples: Vec<f64> = (0..40).map(|i| f64::from(i * 7 % 11) - 5.0).collect();
        let mut out = Vec::new();
        FirDecimator::new(&taps, 2).process(&samples, &mut out);
        let padded: Vec<f64> = core::iter::repeat_n(0.0, taps.len() - 1)
            .chain(samples)
            .collect();
        let want: Vec<f64> = padded
            .windows(taps.len())
            .skip(1)
            .step_by(2)
            .map(|w| w.iter().zip(&taps).map(|(x, t)| x * t).sum())
            .collect();
        assert_eq!(out.len(), want.len());
        for (got, want) in out.iter().zip(&want) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn the_folded_dot_product_is_the_plain_one() {
        let taps = [1.0, -2.0, 3.0, 0.5, 4.0, 0.5, 3.0, -2.0, 1.0];
        let samples: Vec<f64> = (0..taps.len())
            .map(|i| f64::from(i as u8 * 3 % 7) - 2.5)
            .collect();
        let want: f64 = taps.iter().zip(&samples).map(|(t, x)| t * x).sum();
        let mut stage = FirDecimator::new(&taps, 1);
        let mut out = Vec::new();
        stage.process(&samples, &mut out);
        // The last output is the first whose window holds no silence.
        let got = out[out.len() - 1];
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
}
