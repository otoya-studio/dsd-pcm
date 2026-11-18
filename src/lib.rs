//! DSD to PCM conversion.
//!
//! A [`Decimator`] turns a 1-bit DSD stream (DSD64 to DSD1024, in the 44.1 kHz or 48 kHz
//! family) into PCM at the highest rate DSD rate / 2ⁿ that stays at or below a limit,
//! 96 kHz unless set otherwise: DSD64 to DSD512 give 88.2 kHz, the 48 kHz family gives 96 kHz.
//!
//! The conversion is a cascade of linear-phase FIR filters in 64-bit floating point:
//!
//! 1. **÷8 at the DSD rate**, one output per input byte. The filter is evaluated through
//!    lookup tables (8 taps per table, one entry per byte value), so no work is done per bit.
//! 2. **÷2 stages** while more than ÷4 is left (DSD128 and above). They are half-band
//!    filters: every second tap is zero, and the stage skips it.
//! 3. **A final ÷4 low-pass**, flat up to 0.34 of the output rate (30 kHz at 88.2 kHz). It
//!    removes the ultrasonic noise that DSD's noise shaping pushes above the audio band, before
//!    it could fold back into it.
//!
//! Every stage is a Kaiser-windowed sinc with at least [`STOPBAND_DB`] of attenuation over the
//! bands that would alias into the output. Kaiser's formula for the length is an estimate, so
//! each filter's length is checked against its measured response and grown until it meets
//! the attenuation. [`Filter`] chooses the final low-pass's passband.
//!
//! Around the conversion, [`Quantizer`] turns the `f64` output into 8- to 32-bit integers
//! with TPDF dither, and the [`dop`] module packs DSD into DSD over PCM words and back.
//!
//! **Level:** bit 1 maps to +1.0 and bit 0 to −1.0, at unity gain. A 50 % modulated signal,
//! the 0 dB reference level of SACD, comes out at −6 dBFS, so even the highest level SACD
//! permits (+3 dB) has headroom.
//!
//! **Timing:** the cascade's group delay is compensated. Output frame `k` lines up with DSD
//! bit `k · factor`, and a stream of `n` bits per channel gives exactly `n / factor` frames
//! once [`Decimator::finish`] has flushed the filters.
//!
//! # Example
//!
//! ```
//! use dsd_pcm::{BitOrder, Decimator, DSD64_RATE};
//!
//! // A stereo DSD64 stream, its bytes interleaved by channel as in a DSDIFF file.
//! let mut decimator = Decimator::new(DSD64_RATE, 2)?.with_bit_order(BitOrder::MsbFirst);
//! assert_eq!(decimator.output_rate(), 88_200);
//!
//! let idle = [0x69u8; 2 * 4096]; // the DSD silence pattern on both channels
//! let mut pcm = Vec::new();
//! decimator.process_interleaved(&idle, &mut pcm)?;
//! decimator.finish(&mut pcm);
//!
//! // 4096 bytes = 32768 bits per channel, decimated by 32: 1024 stereo frames.
//! assert_eq!(pcm.len(), 2 * 1024);
//! # Ok::<(), dsd_pcm::Error>(())
//! ```

// The library itself never panics on bad input; tests may unwrap.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

mod design;
pub mod dop;
mod quantize;
mod stage;

use std::fmt;

pub use quantize::Quantizer;

use design::{design, halfband, kaiser_len, lowpass, peak_db};
use stage::{ByteFilter, ByteTables, FirDecimator, SILENCE};

/// DSD64: 64 times 44.1 kHz, the rate of SACD.
pub const DSD64_RATE: u32 = 2_822_400;

/// The output rate limit [`Decimator::new`] uses.
pub const DEFAULT_MAX_OUTPUT_RATE: u32 = 96_000;

/// Minimum stopband attenuation of every stage, in dB.
pub const STOPBAND_DB: f64 = 120.0;

/// Upper edge of the final stage's passband, as a fraction of the output rate.
const PASSBAND_FRACTION: f64 = 0.34;

/// The response of a [`Decimator`]'s final low-pass, set with [`Decimator::with_filter`].
///
/// Both are linear-phase with at least [`STOPBAND_DB`] of attenuation over every band that
/// would alias, and both keep the timing exact; they differ in what they let through above
/// the audio band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Filter {
    /// Flat to 0.34 of the output rate (30 kHz at 88.2 kHz), stopband from the Nyquist
    /// frequency. Passes the ultrasonic content of high-rate recordings; with DSD64 it also
    /// passes the lower edge of the noise that noise shaping puts above 20 kHz.
    #[default]
    Wide,
    /// Flat to 20 kHz, stopband from 24 kHz (or from the Nyquist frequency where that is
    /// lower; the passband then ends at 10/11 of it if that is below 20 kHz). Removes DSD's
    /// ultrasonic noise before it reaches amplifiers and tweeters, and at 44.1 or 48 kHz
    /// output it keeps the passband to 20 kHz where [`Filter::Wide`] ends at 15 or 16 kHz.
    /// The steeper filter costs up to about twice the computation of [`Filter::Wide`].
    Audio,
}

impl Filter {
    /// The final stage's passband edge and stopband start in Hz, for an output at `pcm_rate`.
    fn band(self, pcm_rate: f64) -> (f64, f64) {
        let nyquist = pcm_rate / 2.0;
        match self {
            Self::Wide => (PASSBAND_FRACTION * pcm_rate, nyquist),
            Self::Audio => {
                let stop = nyquist.min(24_000.0);
                ((nyquist * 10.0 / 11.0).min(20_000.0), stop)
            }
        }
    }
}

/// The order of the bits within a byte of DSD data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BitOrder {
    /// The most significant bit is the earliest sample (DSDIFF, DSD over PCM, ALSA's `DSD_U8`).
    #[default]
    MsbFirst,
    /// The least significant bit is the earliest sample (DSF).
    LsbFirst,
}

/// Why a [`Decimator`] cannot be built or cannot take some input.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// No power of two of at least 8 divides the DSD rate down to the output rate limit.
    UnsupportedRate {
        /// The DSD rate asked for, in Hz.
        dsd_rate: u32,
        /// The output rate limit, in Hz.
        max_output_rate: u32,
    },
    /// The stream has no channels.
    NoChannels,
    /// The channel slices passed to [`Decimator::process_planar`] differ in number or length.
    ChannelMismatch,
    /// The length of the data passed to [`Decimator::process_interleaved`] is not a multiple
    /// of the channel count.
    IncompleteFrame,
    /// The samples passed to [`dop::DopUnpacker::unpack`] carry no valid DoP markers.
    NotDop,
    /// [`Quantizer::new`] takes word lengths of 8 to 32 bits.
    UnsupportedBits {
        /// The word length asked for.
        bits: u32,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRate {
                dsd_rate,
                max_output_rate,
            } => write!(
                f,
                "a DSD rate of {dsd_rate} Hz cannot be decimated by a power of two to at most \
                 {max_output_rate} Hz"
            ),
            Self::NoChannels => f.write_str("a DSD stream needs at least one channel"),
            Self::ChannelMismatch => {
                f.write_str("expected one slice per channel, all of the same length")
            }
            Self::IncompleteFrame => {
                f.write_str("interleaved data must hold the same number of bytes for every channel")
            }
            Self::NotDop => f.write_str("the samples carry no valid DSD over PCM markers"),
            Self::UnsupportedBits { bits } => {
                write!(
                    f,
                    "a word length of {bits} bits is not supported, only 8 to 32"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// The decimation factor for `dsd_rate`: the smallest power of two of at least 8 that brings it
/// to `max_output_rate` or below, if it divides `dsd_rate` exactly.
///
/// This is what [`Decimator::with_max_output_rate`] uses; a caller can plan the output rate
/// with it before opening a stream.
///
/// ```
/// assert_eq!(dsd_pcm::decimation_factor(2_822_400, 96_000), Some(32));
/// assert_eq!(dsd_pcm::decimation_factor(12_288_000, 96_000), Some(128));
/// assert_eq!(dsd_pcm::decimation_factor(0, 96_000), None);
/// ```
#[must_use]
pub fn decimation_factor(dsd_rate: u32, max_output_rate: u32) -> Option<u32> {
    if dsd_rate == 0 || max_output_rate == 0 {
        return None;
    }
    let mut factor = 8u32;
    while dsd_rate / factor > max_output_rate {
        factor = factor.checked_mul(2)?;
    }
    dsd_rate.is_multiple_of(factor).then_some(factor)
}

/// The filter state of one channel.
#[derive(Debug, Clone)]
struct Channel {
    bytes: ByteFilter,
    stages: Vec<FirDecimator>,
    /// Output of the block being converted.
    out: Vec<f64>,
    /// The other buffer the stages filter between.
    scratch: Vec<f64>,
}

impl Channel {
    /// Converts a block of table indices, one per byte; the output lands in `out`.
    ///
    /// Each stage filters the whole block before the next one starts, so every loop runs
    /// over one filter's state only.
    fn convert(&mut self, indices: impl Iterator<Item = u16>, tables: &ByteTables) {
        let mut block = std::mem::take(&mut self.out);
        block.clear();
        self.bytes.process(indices, tables, &mut block);
        for stage in &mut self.stages {
            self.scratch.clear();
            stage.process(&block, &mut self.scratch);
            std::mem::swap(&mut block, &mut self.scratch);
        }
        self.out = block;
    }

    /// Converts a block of input bytes in `order`.
    fn convert_bytes(
        &mut self,
        bytes: impl Iterator<Item = u8>,
        order: BitOrder,
        tables: &ByteTables,
    ) {
        // One loop per bit order, so the choice is made once and not for every byte.
        match order {
            BitOrder::MsbFirst => self.convert(bytes.map(u16::from), tables),
            BitOrder::LsbFirst => self.convert(bytes.map(|b| u16::from(b.reverse_bits())), tables),
        }
    }
}

/// Bytes per channel converted in one pass: each channel runs through a block on its own,
/// which keeps its filter state in cache, and the outputs are interleaved afterwards.
const BLOCK: usize = 4096;

/// Converts a DSD stream of one or more channels to PCM.
///
/// Feed the stream with [`process_interleaved`](Self::process_interleaved) or
/// [`process_planar`](Self::process_planar), call [`finish`](Self::finish) at its end, and
/// [`reset`](Self::reset) before a seek. Output is appended to a `Vec<f64>` as interleaved
/// frames at [`output_rate`](Self::output_rate), with full scale at ±1.0.
///
/// Input is converted in blocks of a few thousand bytes per channel, so memory use does not
/// grow with the size of a call: it depends only on the filter lengths and the block size.
/// How the input is cut into calls does not change the output.
#[derive(Debug, Clone)]
pub struct Decimator {
    dsd_rate: u32,
    factor: u32,
    filter: Filter,
    order: BitOrder,
    tables: ByteTables,
    channels: Vec<Channel>,
    /// The state of a channel before its first byte; [`Decimator::reset`] restores it.
    fresh: Channel,
    /// Output frames that are still to be dropped at the start (the group delay).
    trim: usize,
    /// The group delay in output frames: dropped at the start, flushed at the end.
    delay: usize,
}

impl Decimator {
    /// A decimator for `channels` channels at `dsd_rate` Hz, with an output rate of at most
    /// [`DEFAULT_MAX_OUTPUT_RATE`] (96 kHz).
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedRate`] if no power of two divides `dsd_rate` down to 96 kHz,
    /// [`Error::NoChannels`] if `channels` is zero.
    pub fn new(dsd_rate: u32, channels: usize) -> Result<Self, Error> {
        Self::with_max_output_rate(dsd_rate, channels, DEFAULT_MAX_OUTPUT_RATE)
    }

    /// A decimator whose output rate is the highest `dsd_rate / 2ⁿ` (n ≥ 3) at or below
    /// `max_output_rate`.
    ///
    /// A lower limit costs less to compute: DSD64 with a limit of 48 kHz gives 44.1 kHz.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedRate`] if [`decimation_factor`] finds no factor,
    /// [`Error::NoChannels`] if `channels` is zero.
    pub fn with_max_output_rate(
        dsd_rate: u32,
        channels: usize,
        max_output_rate: u32,
    ) -> Result<Self, Error> {
        if channels == 0 {
            return Err(Error::NoChannels);
        }
        let factor =
            decimation_factor(dsd_rate, max_output_rate).ok_or(Error::UnsupportedRate {
                dsd_rate,
                max_output_rate,
            })?;
        Ok(Self::build(dsd_rate, channels, factor, Filter::Wide))
    }

    /// Designs the cascade for `dsd_rate / factor` with `filter` as the final low-pass.
    fn build(dsd_rate: u32, channels: usize, factor: u32, filter: Filter) -> Self {
        let pcm_rate = f64::from(dsd_rate / factor);
        let nyquist = pcm_rate / 2.0;

        // Stages 2 to n: ÷2 while more than ÷4 is left, then the final low-pass. Each ÷2 stage
        // only has to keep its own images out of [0, Nyquist of the output].
        let rate1 = f64::from(dsd_rate) / 8.0;
        let mut rate = rate1;
        let mut remaining = (factor / 8) as usize;
        let mut delay_bits = 0.0;
        let mut bits_per_sample = 8.0;
        let mut stages = Vec::new();
        while remaining > 4 {
            let stage = FirDecimator::new(&halfband(rate, nyquist, STOPBAND_DB), 2);
            delay_bits += (stage.len() - 1) as f64 / 2.0 * bits_per_sample;
            stages.push(stage);
            rate /= 2.0;
            bits_per_sample *= 2.0;
            remaining /= 2;
        }
        let (pass, stop) = filter.band(pcm_rate);
        let last = FirDecimator::new(&design(rate, pass, stop, STOPBAND_DB), remaining);
        delay_bits += (last.len() - 1) as f64 / 2.0 * bits_per_sample;
        stages.push(last);

        // Output k is computed once bit (k + 1) · factor − 1 has arrived, and shows the input
        // delayed by the cascade's group delay; dropping the first `trim` outputs makes it show
        // bit k · factor. Stage 1 must be a multiple of 8 taps long and keep the images around
        // multiples of its output rate, which fold onto [0, Nyquist], at least STOPBAND_DB
        // down. Of the next lengths that do, the one that leaves the smallest fraction of a
        // sample of misalignment is taken.
        let dsd = f64::from(dsd_rate);
        let cutoff = rate1 / 2.0 / dsd;
        let images: Vec<(f64, f64)> = (1..)
            .map(|k| f64::from(k) * rate1)
            .take_while(|centre| centre - nyquist < dsd / 2.0)
            .map(|centre| {
                (
                    (centre - nyquist) / dsd,
                    ((centre + nyquist) / dsd).min(0.5),
                )
            })
            .collect();
        let factor_f = f64::from(factor);
        let min_len = kaiser_len(STOPBAND_DB, (rate1 - 2.0 * nyquist) / dsd).next_multiple_of(8);
        let stage1 = (0..16)
            .map(|extra| lowpass(min_len + 8 * extra, cutoff, STOPBAND_DB))
            .filter(|taps| peak_db(taps, &images) <= -STOPBAND_DB)
            .take(8)
            .map(|taps| {
                let total = delay_bits + (taps.len() - 1) as f64 / 2.0 - (factor_f - 1.0);
                let trim = (total / factor_f).round().max(0.0);
                let misalignment = (total - trim * factor_f).abs();
                (taps, trim as usize, misalignment)
            })
            .min_by(|a, b| a.2.total_cmp(&b.2));
        // Some length within 16 steps always meets the attenuation; the fallback only keeps
        // the code free of panics.
        let (stage1, trim) = stage1.map_or_else(
            || (lowpass(min_len + 8 * 16, cutoff, STOPBAND_DB), 0),
            |(taps, trim, _)| (taps, trim),
        );
        let tables = ByteTables::new(&stage1);
        let fresh = Channel {
            bytes: ByteFilter::new(tables.len()),
            stages,
            out: Vec::new(),
            scratch: Vec::new(),
        };
        Self {
            dsd_rate,
            factor,
            filter,
            order: BitOrder::MsbFirst,
            channels: vec![fresh.clone(); channels],
            fresh,
            tables,
            trim,
            delay: trim,
        }
    }

    /// Uses `filter` as the final low-pass ([`Filter::Wide`] unless set).
    ///
    /// The cascade is designed anew, so set the filter before converting: the decimator
    /// starts from a fresh stream, keeping its rates, channels and bit order.
    #[must_use]
    pub fn with_filter(self, filter: Filter) -> Self {
        if filter == self.filter {
            return self;
        }
        let order = self.order;
        Self::build(self.dsd_rate, self.channels.len(), self.factor, filter).with_bit_order(order)
    }

    /// The final low-pass in use.
    #[must_use]
    pub fn filter(&self) -> Filter {
        self.filter
    }

    /// Sets the bit order of the input bytes (MSB first unless set).
    #[must_use]
    pub fn with_bit_order(mut self, order: BitOrder) -> Self {
        self.order = order;
        self
    }

    /// The DSD rate in Hz.
    #[must_use]
    pub fn dsd_rate(&self) -> u32 {
        self.dsd_rate
    }

    /// The PCM output rate in Hz.
    #[must_use]
    pub fn output_rate(&self) -> u32 {
        self.dsd_rate / self.factor
    }

    /// DSD bits per channel per output frame.
    #[must_use]
    pub fn factor(&self) -> u32 {
        self.factor
    }

    /// Number of channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    /// Output frames for `bits` DSD bits per channel, as a complete stream gives them.
    #[must_use]
    pub fn output_frames(&self, bits: u64) -> u64 {
        bits / u64::from(self.factor)
    }

    /// The latency of the cascade in output frames: how many frames of output trail the
    /// input. [`finish`](Self::finish) emits them.
    #[must_use]
    pub fn latency(&self) -> usize {
        self.delay
    }

    /// Converts bytes interleaved by channel: one byte of channel 0, one of channel 1, and so
    /// on, as stored in DSDIFF files.
    ///
    /// # Errors
    ///
    /// [`Error::IncompleteFrame`] if `bytes` does not hold the same number of bytes for every
    /// channel; nothing is converted then.
    pub fn process_interleaved(&mut self, bytes: &[u8], out: &mut Vec<f64>) -> Result<(), Error> {
        let channels = self.channels.len();
        if !bytes.len().is_multiple_of(channels) {
            return Err(Error::IncompleteFrame);
        }
        for block in bytes.chunks(BLOCK * channels) {
            for (c, channel) in self.channels.iter_mut().enumerate() {
                let own = block.iter().skip(c).step_by(channels).copied();
                channel.convert_bytes(own, self.order, &self.tables);
            }
            self.emit(out);
        }
        Ok(())
    }

    /// Converts one slice of bytes per channel, all of the same length, as stored in the
    /// blocks of a DSF file.
    ///
    /// # Errors
    ///
    /// [`Error::ChannelMismatch`] if the number of slices is not the channel count or their
    /// lengths differ; nothing is converted then.
    pub fn process_planar<B: AsRef<[u8]>>(
        &mut self,
        channels: &[B],
        out: &mut Vec<f64>,
    ) -> Result<(), Error> {
        let len = channels.first().map_or(0, |c| c.as_ref().len());
        if channels.len() != self.channels.len() || channels.iter().any(|c| c.as_ref().len() != len)
        {
            return Err(Error::ChannelMismatch);
        }
        for start in (0..len).step_by(BLOCK) {
            let end = (start + BLOCK).min(len);
            for (channel, bytes) in self.channels.iter_mut().zip(channels) {
                let own = bytes.as_ref()[start..end].iter().copied();
                channel.convert_bytes(own, self.order, &self.tables);
            }
            self.emit(out);
        }
        Ok(())
    }

    /// Ends the stream: feeds silence through the filters until the last input bit has
    /// reached the output, so a stream of `n` bits gives exactly
    /// [`output_frames(n)`](Self::output_frames) frames.
    ///
    /// The decimator is [`reset`](Self::reset) afterwards and can take a new stream.
    pub fn finish(&mut self, out: &mut Vec<f64>) {
        let bytes = self.delay * (self.factor as usize / 8);
        for start in (0..bytes).step_by(BLOCK) {
            let count = (start + BLOCK).min(bytes) - start;
            for channel in &mut self.channels {
                channel.convert(std::iter::repeat_n(SILENCE, count), &self.tables);
            }
            self.emit(out);
        }
        self.reset();
    }

    /// Forgets the stream so far, as if nothing had been converted. Call it before converting
    /// from a new position after a seek: the output then starts with the frame for the first
    /// new bit, with the filters started from silence.
    pub fn reset(&mut self) {
        for channel in &mut self.channels {
            channel.clone_from(&self.fresh);
        }
        self.trim = self.delay;
    }

    /// Interleaves the channels' outputs of the last block into `out`, after dropping what is
    /// left of the group delay. Every channel took the same bytes, so each has as many outputs.
    fn emit(&mut self, out: &mut Vec<f64>) {
        let frames = self.channels[0].out.len();
        let skip = self.trim.min(frames);
        self.trim -= skip;
        out.reserve((frames - skip) * self.channels.len());
        for frame in skip..frames {
            out.extend(self.channels.iter().map(|channel| channel.out[frame]));
        }
        for channel in &mut self.channels {
            channel.out.clear();
        }
    }
}

// The README's examples run as doctests, so they stay correct.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
