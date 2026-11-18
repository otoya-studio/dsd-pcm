//! DSD over PCM (DoP), version 1.1: DSD carried in 24-bit PCM samples.
//!
//! Each PCM sample holds 16 DSD bits of one channel in its low 16 bits, the earlier byte in
//! the higher one and each byte MSB first, and a marker in its top byte. The marker is
//! [`MARKERS`]`[0]` (0x05) in every channel of one frame and [`MARKERS`]`[1]` (0xFA) in the
//! next, alternating, so a DAC can tell DoP from PCM. The PCM rate is the DSD rate / 16
//! ([`pcm_rate`]): DSD64 travels as 176.4 kHz.
//!
//! Samples are `u32` with the 24-bit word in the low 24 bits; for a 32-bit interface, shift
//! them left by 8. [`DopPacker`] makes them from DSD bytes, [`DopUnpacker`] turns them back
//! into DSD bytes that [`Decimator`](crate::Decimator) takes.
//!
//! ```
//! use dsd_pcm::dop::{DopPacker, DopUnpacker};
//!
//! let dsd = [0x69u8; 2 * 64]; // stereo, bytes interleaved by channel
//! let mut words = Vec::new();
//! DopPacker::new(2)?.pack_interleaved(&dsd, &mut words)?;
//! assert_eq!(words.len(), 64); // 2 bytes per channel per word
//! assert_eq!(words[0] >> 16, 0x05);
//!
//! let mut back = Vec::new();
//! DopUnpacker::new(2)?.unpack(&words, &mut back)?;
//! assert_eq!(back, dsd);
//! # Ok::<(), dsd_pcm::Error>(())
//! ```

use crate::{BitOrder, Error};

/// The two marker bytes, in the order they alternate from frame to frame.
pub const MARKERS: [u8; 2] = [0x05, 0xFA];

/// Frames [`detect`] wants to see before it calls a stream DoP.
pub const DETECT_FRAMES: usize = 16;

/// The PCM rate that carries `dsd_rate` as DoP: 16 DSD bits per sample.
#[must_use]
pub const fn pcm_rate(dsd_rate: u32) -> u32 {
    dsd_rate / 16
}

/// Whether `samples` (interleaved, `channels` per frame) start with at least
/// [`DETECT_FRAMES`] frames of valid DoP markers.
///
/// Music in 24-bit PCM practically never repeats this pattern over 16 frames; fewer samples
/// give `false`.
#[must_use]
pub fn detect(samples: &[u32], channels: usize) -> bool {
    channels > 0
        && samples.len() >= DETECT_FRAMES * channels
        && frames_valid(&samples[..DETECT_FRAMES * channels], channels).is_some()
}

/// The marker index of the first frame if every frame of `samples` carries one marker in all
/// its channels and the markers alternate; `None` otherwise or if `samples` is empty.
fn frames_valid(samples: &[u32], channels: usize) -> Option<usize> {
    let first = samples.first().map(|&s| marker(s))?;
    let start = MARKERS.iter().position(|&m| m == first)?;
    samples
        .chunks(channels)
        .enumerate()
        .all(|(frame, words)| {
            let expected = MARKERS[(start + frame) % 2];
            words.iter().all(|&w| marker(w) == expected)
        })
        .then_some(start)
}

/// The marker byte of a 24-bit DoP word.
fn marker(word: u32) -> u8 {
    // Bits 16 to 23; the cast keeps exactly those.
    (word >> 16) as u8
}

/// One DoP word from a marker and two DSD bytes, the earlier first.
fn word(marker: u8, first: u8, second: u8) -> u32 {
    u32::from(marker) << 16 | u32::from(first) << 8 | u32::from(second)
}

/// Packs DSD bytes into DoP words.
///
/// The markers keep alternating across calls; [`reset`](Self::reset) starts again with
/// 0x05.
#[derive(Debug, Clone)]
pub struct DopPacker {
    channels: usize,
    order: BitOrder,
    next: usize,
}

impl DopPacker {
    /// A packer for `channels` channels whose input bytes are MSB first.
    ///
    /// # Errors
    ///
    /// [`Error::NoChannels`] if `channels` is zero.
    pub fn new(channels: usize) -> Result<Self, Error> {
        if channels == 0 {
            return Err(Error::NoChannels);
        }
        Ok(Self {
            channels,
            order: BitOrder::MsbFirst,
            next: 0,
        })
    }

    /// Sets the bit order of the input bytes; DoP itself is always MSB first, so LSB-first
    /// bytes (DSF) are reversed on the way.
    #[must_use]
    pub fn with_bit_order(mut self, order: BitOrder) -> Self {
        self.order = order;
        self
    }

    /// Packs bytes interleaved by channel (one byte of channel 0, one of channel 1, …), two
    /// bytes per channel per word, and appends the words interleaved by channel.
    ///
    /// # Errors
    ///
    /// [`Error::IncompleteFrame`] if `bytes` does not hold two bytes per channel for every
    /// frame; nothing is packed then.
    pub fn pack_interleaved(&mut self, bytes: &[u8], out: &mut Vec<u32>) -> Result<(), Error> {
        let frame = 2 * self.channels;
        if !bytes.len().is_multiple_of(frame) {
            return Err(Error::IncompleteFrame);
        }
        out.reserve(bytes.len() / 2);
        for pair in bytes.chunks_exact(frame) {
            let marker = self.marker();
            let (first, second) = pair.split_at(self.channels);
            out.extend(
                first
                    .iter()
                    .zip(second)
                    .map(|(&a, &b)| word(marker, self.msb(a), self.msb(b))),
            );
        }
        Ok(())
    }

    /// Packs one slice of bytes per channel, all of the same even length, and appends the
    /// words interleaved by channel.
    ///
    /// # Errors
    ///
    /// [`Error::ChannelMismatch`] if the number of slices is not the channel count, their
    /// lengths differ or are odd; nothing is packed then.
    pub fn pack_planar<B: AsRef<[u8]>>(
        &mut self,
        channels: &[B],
        out: &mut Vec<u32>,
    ) -> Result<(), Error> {
        let len = channels.first().map_or(0, |c| c.as_ref().len());
        if channels.len() != self.channels
            || !len.is_multiple_of(2)
            || channels.iter().any(|c| c.as_ref().len() != len)
        {
            return Err(Error::ChannelMismatch);
        }
        out.reserve(len / 2 * self.channels);
        for i in (0..len).step_by(2) {
            let marker = self.marker();
            out.extend(channels.iter().map(|c| {
                let c = c.as_ref();
                word(marker, self.msb(c[i]), self.msb(c[i + 1]))
            }));
        }
        Ok(())
    }

    /// Starts the marker sequence again with 0x05, for a new stream.
    pub fn reset(&mut self) {
        self.next = 0;
    }

    /// The marker of the next frame; advances the sequence.
    fn marker(&mut self) -> u8 {
        let marker = MARKERS[self.next];
        self.next ^= 1;
        marker
    }

    /// `byte` in MSB-first order.
    fn msb(&self, byte: u8) -> u8 {
        match self.order {
            BitOrder::MsbFirst => byte,
            BitOrder::LsbFirst => byte.reverse_bits(),
        }
    }
}

/// Unpacks DoP words into DSD bytes, checking the markers.
///
/// The marker sequence is followed across calls; [`reset`](Self::reset) forgets it, for a
/// new stream or after a seek.
#[derive(Debug, Clone)]
pub struct DopUnpacker {
    channels: usize,
    /// The marker index the next frame must carry, once a frame has been seen.
    next: Option<usize>,
}

impl DopUnpacker {
    /// An unpacker for `channels` channels.
    ///
    /// # Errors
    ///
    /// [`Error::NoChannels`] if `channels` is zero.
    pub fn new(channels: usize) -> Result<Self, Error> {
        if channels == 0 {
            return Err(Error::NoChannels);
        }
        Ok(Self {
            channels,
            next: None,
        })
    }

    /// Unpacks words interleaved by channel and appends the DSD bytes interleaved by
    /// channel, MSB first (as DSDIFF stores them and [`Decimator::process_interleaved`]
    /// takes them by default). Only the low 24 bits of each word are read.
    ///
    /// # Errors
    ///
    /// [`Error::IncompleteFrame`] if `samples` is not a whole number of frames,
    /// [`Error::NotDop`] if a frame's markers are missing, differ between channels or do not
    /// alternate; nothing is unpacked then.
    ///
    /// [`Decimator::process_interleaved`]: crate::Decimator::process_interleaved
    pub fn unpack(&mut self, samples: &[u32], out: &mut Vec<u8>) -> Result<(), Error> {
        if !samples.len().is_multiple_of(self.channels) {
            return Err(Error::IncompleteFrame);
        }
        let Some(start) = frames_valid(samples, self.channels) else {
            return if samples.is_empty() {
                Ok(())
            } else {
                Err(Error::NotDop)
            };
        };
        if self.next.is_some_and(|next| next != start) {
            return Err(Error::NotDop);
        }
        let frames = samples.len() / self.channels;
        self.next = Some((start + frames) % 2);
        out.reserve(2 * samples.len());
        for frame in samples.chunks_exact(self.channels) {
            // The cast keeps bits 8 to 15 (earlier byte) and 0 to 7 (later byte).
            out.extend(frame.iter().map(|&w| (w >> 8) as u8));
            out.extend(frame.iter().map(|&w| w as u8));
        }
        Ok(())
    }

    /// Forgets the marker sequence, for a new stream or after a seek.
    pub fn reset(&mut self) {
        self.next = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_carry_the_marker_and_both_bytes() {
        let mut words = Vec::new();
        let mut packer = DopPacker::new(1).unwrap();
        packer
            .pack_interleaved(&[0x12, 0x34, 0x56, 0x78], &mut words)
            .unwrap();
        assert_eq!(words, [0x05_1234, 0xFA_5678]);
        packer.pack_interleaved(&[0x9A, 0xBC], &mut words).unwrap();
        assert_eq!(words[2], 0x05_9ABC, "markers alternate across calls");
    }

    #[test]
    fn planar_and_interleaved_input_give_the_same_words() {
        let left = [1u8, 2, 3, 4];
        let right = [5u8, 6, 7, 8];
        let interleaved = [1u8, 5, 2, 6, 3, 7, 4, 8];
        let (mut a, mut b) = (Vec::new(), Vec::new());
        DopPacker::new(2)
            .unwrap()
            .pack_planar(&[left, right], &mut a)
            .unwrap();
        DopPacker::new(2)
            .unwrap()
            .pack_interleaved(&interleaved, &mut b)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(a, [0x05_0102, 0x05_0506, 0xFA_0304, 0xFA_0708]);
    }

    #[test]
    fn lsb_first_input_is_reversed() {
        let mut words = Vec::new();
        DopPacker::new(1)
            .unwrap()
            .with_bit_order(BitOrder::LsbFirst)
            .pack_interleaved(&[0x01, 0x80], &mut words)
            .unwrap();
        assert_eq!(words, [0x05_8001]);
    }

    #[test]
    fn unpacking_checks_the_markers() {
        let mut out = Vec::new();
        let mut unpacker = DopUnpacker::new(2).unwrap();
        assert_eq!(
            unpacker.unpack(&[0x05_0000, 0xFA_0000], &mut out),
            Err(Error::NotDop)
        );
        assert_eq!(
            unpacker.unpack(&[0x05_0000, 0x05_0000, 0x05_0000, 0x05_0000], &mut out),
            Err(Error::NotDop)
        );
        assert_eq!(
            unpacker.unpack(&[0x12_0000, 0x12_0000], &mut out),
            Err(Error::NotDop)
        );
        assert_eq!(
            unpacker.unpack(&[0x05_0000], &mut out),
            Err(Error::IncompleteFrame)
        );
        assert!(out.is_empty());
        unpacker.unpack(&[0xFA_0102, 0xFA_0304], &mut out).unwrap();
        assert_eq!(out, [1, 3, 2, 4]);
        assert_eq!(
            unpacker.unpack(&[0xFA_0000, 0xFA_0000], &mut out),
            Err(Error::NotDop),
            "the sequence continues across calls"
        );
        unpacker.unpack(&[0x05_0000, 0x05_0000], &mut out).unwrap();
        // The top byte of a 32-bit word is ignored.
        unpacker
            .unpack(&[0xFFFA_0000, 0x00FA_0000], &mut out)
            .unwrap();
    }

    #[test]
    fn detection_needs_sixteen_valid_frames() {
        let mut words = Vec::new();
        DopPacker::new(2)
            .unwrap()
            .pack_interleaved(&[0x69; 64], &mut words)
            .unwrap();
        assert!(detect(&words, 2));
        assert!(!detect(&words[..30], 2));
        words[7] = 0x12_3456;
        assert!(!detect(&words, 2));
        assert!(!detect(&[0; 64], 2));
        assert!(!detect(&words, 0));
    }

    #[test]
    fn the_pcm_rate_is_a_sixteenth() {
        assert_eq!(pcm_rate(2_822_400), 176_400);
    }
}
