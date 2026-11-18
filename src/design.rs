//! Kaiser-windowed sinc low-pass filters.
//!
//! The formulas for the window's shape parameter and the filter length are Kaiser's empirical
//! fits, as given in Oppenheim & Schafer, *Discrete-Time Signal Processing*, section 7.5.3.

use std::f64::consts::PI;

/// Modified Bessel function of the first kind, order zero, by its power series.
///
/// The series converges for every real `x`; it stops once a term no longer changes the sum
/// in double precision.
fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..500 {
        let f = half / f64::from(k);
        term *= f * f;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// Kaiser's shape parameter β for a stopband attenuation of `atten_db`.
fn kaiser_beta(atten_db: f64) -> f64 {
    if atten_db > 50.0 {
        0.1102 * (atten_db - 8.7)
    } else if atten_db > 21.0 {
        0.5842 * (atten_db - 21.0).powf(0.4) + 0.07886 * (atten_db - 21.0)
    } else {
        0.0
    }
}

/// Number of taps a Kaiser low-pass needs for `atten_db` over a transition band of
/// `transition` (as a fraction of the sample rate); at least 3.
pub(crate) fn kaiser_len(atten_db: f64, transition: f64) -> usize {
    let order = (atten_db - 7.95) / (2.285 * 2.0 * PI * transition);
    (order.ceil().max(2.0) as usize) + 1
}

/// A linear-phase low-pass of `len` taps with its -6 dB point at `cutoff` (a fraction of the
/// sample rate), normalised to unity gain at DC.
pub(crate) fn lowpass(len: usize, cutoff: f64, atten_db: f64) -> Vec<f64> {
    let len = len.max(3);
    let beta = kaiser_beta(atten_db);
    let middle = (len - 1) as f64 / 2.0;
    let norm = bessel_i0(beta);
    let mut taps: Vec<f64> = (0..len)
        .map(|i| {
            let t = i as f64 - middle;
            let x = 2.0 * cutoff * t;
            let sinc = if x == 0.0 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let r = t / middle;
            let window = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / norm;
            sinc * window
        })
        .collect();
    // The two halves are equal in exact arithmetic; copying one onto the other makes them
    // equal in floating point too, so the filter is exactly linear-phase.
    for i in 0..len / 2 {
        taps[len - 1 - i] = taps[i];
    }
    let sum: f64 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= sum;
    }
    taps
}

/// A low-pass for a signal at `rate` Hz that passes up to `pass` Hz and attenuates by at least
/// `atten_db` from `stop` Hz to `rate / 2`.
///
/// The Kaiser formula for the length is an estimate that falls a few dB short for short
/// filters, so the length grows until the measured response meets the attenuation.
pub(crate) fn design(rate: f64, pass: f64, stop: f64, atten_db: f64) -> Vec<f64> {
    let cutoff = pass.midpoint(stop) / rate;
    let band = [(stop / rate, 0.5)];
    let mut len = kaiser_len(atten_db, (stop - pass) / rate);
    loop {
        let taps = lowpass(len, cutoff, atten_db);
        if peak_db(&taps, &band) <= -atten_db {
            return taps;
        }
        len += 1;
    }
}

/// A half-band low-pass for a signal at `rate` Hz that halves the rate: it passes up to `pass`
/// Hz and attenuates by at least `atten_db` from `rate / 2 − pass` to `rate / 2`, the band that
/// folds onto the passband.
///
/// Its cutoff is a quarter of the rate, so every second tap away from the centre is zero: the
/// length is 4k + 3, where the outermost taps are not among them, and the zeros are set exactly
/// so the stage can skip them. As in [`design`], the length grows until the measured response
/// meets the attenuation.
pub(crate) fn halfband(rate: f64, pass: f64, atten_db: f64) -> Vec<f64> {
    let band = [((rate / 2.0 - pass) / rate, 0.5)];
    let mut len =
        (kaiser_len(atten_db, (rate / 2.0 - 2.0 * pass) / rate) + 1).next_multiple_of(4) - 1;
    loop {
        let mut taps = lowpass(len, 0.25, atten_db);
        let middle = len / 2;
        for (i, tap) in taps.iter_mut().enumerate() {
            if i != middle && i.abs_diff(middle) % 2 == 0 {
                *tap = 0.0;
            }
        }
        if peak_db(&taps, &band) <= -atten_db {
            return taps;
        }
        len += 4;
    }
}

/// The highest magnitude response of `taps`, in dB, over `bands` (pairs of frequencies as
/// fractions of the sample rate).
///
/// Each band is sampled at 16 points per sidelobe of the filter (sidelobes are about
/// 1 / length wide), enough to find each lobe's peak to within a small fraction of a dB.
pub(crate) fn peak_db(taps: &[f64], bands: &[(f64, f64)]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    for &(low, high) in bands {
        let points = ((high - low) * taps.len() as f64 * 16.0).ceil().max(64.0) as usize;
        for step in 0..=points {
            let freq = low + (high - low) * step as f64 / points as f64;
            peak = peak.max(response_db(taps, freq));
        }
    }
    peak
}

/// The magnitude response of the linear-phase (symmetric) filter `taps` at `freq` (a fraction
/// of the sample rate), in dB.
///
/// A symmetric filter's response is its delay times the real amplitude
/// `Σ h[n] · cos(ω (n − M))`, with `M` the middle; the cosines come from the recurrence
/// `cos(x + ω) = 2 cos ω cos x − cos(x − ω)` instead of one library call per tap, which keeps
/// the length search in [`design`] and the stage-1 choice fast. For the few hundred taps used
/// here the recurrence's rounding error stays below 1e-10, far under the −120 dB (1e-6) it
/// has to resolve.
pub(crate) fn response_db(taps: &[f64], freq: f64) -> f64 {
    let omega = 2.0 * PI * freq;
    let middle = (taps.len() as f64 - 1.0) / 2.0;
    let step = 2.0 * omega.cos();
    let mut previous = (omega * (-1.0 - middle)).cos();
    let mut current = (omega * -middle).cos();
    let mut amplitude = 0.0;
    for tap in taps {
        amplitude += tap * current;
        let next = step * current - previous;
        previous = current;
        current = next;
    }
    20.0 * amplitude.abs().log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_matches_reference_values() {
        // Abramowitz & Stegun, table 9.8.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008).abs() < 1e-12);
        assert!((bessel_i0(5.0) - 27.239_871_823_604_44).abs() < 1e-9);
    }

    #[test]
    fn the_fast_response_matches_a_direct_evaluation() {
        for taps in [
            lowpass(31, 0.25, 120.0),
            lowpass(136, 0.0156, 120.0),
            lowpass(197, 0.1, 120.0),
        ] {
            for step in 0..=500 {
                let freq = f64::from(step) / 1000.0;
                let (mut re, mut im) = (0.0, 0.0);
                for (n, tap) in taps.iter().enumerate() {
                    let phase = -2.0 * PI * freq * n as f64;
                    re += tap * phase.cos();
                    im += tap * phase.sin();
                }
                let direct = re.hypot(im);
                let fast = 10f64.powf(response_db(&taps, freq) / 20.0);
                assert!(
                    (direct - fast).abs() < 1e-11,
                    "{} taps at {freq}: {direct} vs {fast}",
                    taps.len()
                );
            }
        }
    }

    #[test]
    fn a_designed_filter_meets_its_specification() {
        let rate = 352_800.0;
        let taps = design(rate, 20_000.0, 44_100.0, 120.0);
        assert!(response_db(&taps, 0.0).abs() < 1e-9, "unity gain at DC");
        for hz in [1_000.0, 10_000.0, 20_000.0] {
            let db = response_db(&taps, hz / rate);
            assert!(db.abs() < 0.001, "flat passband at {hz} Hz: {db} dB");
        }
        for hz in [44_100.0, 60_000.0, 100_000.0, 176_000.0] {
            let db = response_db(&taps, hz / rate);
            assert!(db < -119.0, "stopband at {hz} Hz: {db} dB");
        }
    }

    #[test]
    fn a_half_band_filter_meets_its_specification() {
        for rate in [705_600.0, 1_411_200.0, 2_822_400.0] {
            let taps = halfband(rate, 44_100.0, 120.0);
            assert_eq!(taps.len() % 4, 3);
            let middle = taps.len() / 2;
            for (i, tap) in taps.iter().enumerate() {
                let zero = i != middle && i.abs_diff(middle) % 2 == 0;
                assert_eq!(*tap == 0.0, zero, "{rate} Hz, tap {i}");
            }
            assert!(response_db(&taps, 0.0).abs() < 1e-9, "unity gain at DC");
            for hz in [1_000.0, 20_000.0, 44_100.0] {
                let db = response_db(&taps, hz / rate);
                assert!(
                    db.abs() < 0.001,
                    "{rate} Hz: flat passband at {hz} Hz: {db} dB"
                );
            }
            let stop = rate / 2.0 - 44_100.0;
            for step in 0..=20 {
                let hz = stop + (rate / 2.0 - stop) * f64::from(step) / 20.0;
                let db = response_db(&taps, hz / rate);
                assert!(db < -119.0, "{rate} Hz: stopband at {hz} Hz: {db} dB");
            }
        }
    }
}
