//! Properties that must hold for any input: random data, cut at random places, at every rate
//! and in every layout. A small seeded generator keeps the runs reproducible without a
//! dependency.

use dsd_pcm::{BitOrder, Decimator, Error, DSD64_RATE};

/// xorshift64*: enough randomness for test data, the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A value in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

/// The rates and channel counts the cases draw from.
const RATES: [u32; 5] = [
    DSD64_RATE,
    2 * DSD64_RATE,
    4 * DSD64_RATE,
    3_072_000,
    6_144_000,
];

#[test]
fn any_cut_of_any_input_gives_the_same_output() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for case in 0..24 {
        let rate = RATES[rng.below(RATES.len())];
        let channels = 1 + rng.below(6);
        let order = if rng.below(2) == 0 {
            BitOrder::MsbFirst
        } else {
            BitOrder::LsbFirst
        };
        let len = rng.below(20_000);
        let data = rng.bytes(len * channels);
        let new = || {
            Decimator::new(rate, channels)
                .unwrap()
                .with_bit_order(order)
        };

        let mut whole = Vec::new();
        let mut decimator = new();
        decimator.process_interleaved(&data, &mut whole).unwrap();
        decimator.finish(&mut whole);
        assert_eq!(
            whole.len() as u64,
            channels as u64 * decimator.output_frames(8 * len as u64)
        );

        // The same frames, cut at random frame boundaries, some pieces empty.
        let mut pieces = Vec::new();
        let mut decimator = new();
        let mut start = 0;
        while start < len {
            let end = (start + rng.below(3_000)).min(len);
            decimator
                .process_interleaved(&data[start * channels..end * channels], &mut pieces)
                .unwrap();
            start = end;
        }
        decimator.finish(&mut pieces);
        assert!(
            whole == pieces,
            "case {case}: {rate} Hz, {channels} channels, {len} bytes"
        );

        // Planar input of the same data agrees with interleaved input.
        let planes: Vec<Vec<u8>> = (0..channels)
            .map(|c| data.iter().skip(c).step_by(channels).copied().collect())
            .collect();
        let mut planar = Vec::new();
        let mut decimator = new();
        decimator.process_planar(&planes, &mut planar).unwrap();
        decimator.finish(&mut planar);
        assert!(whole == planar, "case {case}: planar differs");
    }
}

#[test]
fn every_output_is_finite_and_within_the_modulation_limit() {
    // A DSD stream of all ones is the largest signal there is: +1.0 after the filters settle.
    // The ringing of the filters can overshoot it a little, never by much.
    let mut rng = Rng(7);
    for rate in RATES {
        for data in [vec![0xFF; 50_000], vec![0x00; 50_000], rng.bytes(50_000)] {
            let mut out = Vec::new();
            let mut decimator = Decimator::new(rate, 1).unwrap();
            decimator.process_interleaved(&data, &mut out).unwrap();
            decimator.finish(&mut out);
            for v in out {
                assert!(v.is_finite() && v.abs() < 1.2, "{rate} Hz: {v}");
            }
        }
    }
}

#[test]
fn bad_input_is_an_error_never_a_panic() {
    let mut rng = Rng(42);
    for _ in 0..200 {
        let channels = 1 + rng.below(4);
        let mut decimator = Decimator::new(DSD64_RATE, channels).unwrap();
        let mut out = Vec::new();
        let len = rng.below(64);
        let data = rng.bytes(len);
        let interleaved = decimator.process_interleaved(&data, &mut out);
        assert_eq!(
            interleaved.is_ok(),
            data.len().is_multiple_of(channels),
            "{} bytes, {channels} channels",
            data.len()
        );
        let count = rng.below(5);
        let planes: Vec<Vec<u8>> = (0..count)
            .map(|_| {
                let len = rng.below(8);
                rng.bytes(len)
            })
            .collect();
        let fits = planes.len() == channels && planes.iter().all(|p| p.len() == planes[0].len());
        assert_eq!(
            decimator.process_planar(&planes, &mut out).err(),
            (!fits).then_some(Error::ChannelMismatch)
        );
    }
    for rate in [0, 1, 7, 44_100, 2_822_401, u32::MAX] {
        assert!(Decimator::new(rate, 2).is_err(), "{rate} Hz");
    }
}

#[test]
fn a_decimator_can_move_to_another_thread() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Decimator>();
    assert_send_sync::<Error>();
    assert_send_sync::<dsd_pcm::Quantizer>();
    assert_send_sync::<dsd_pcm::dop::DopPacker>();
    assert_send_sync::<dsd_pcm::dop::DopUnpacker>();
}
