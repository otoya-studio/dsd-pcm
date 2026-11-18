//! Converts a DSF file to a 32-bit float WAV file.
//!
//! ```text
//! cargo run --release --example dsf_to_wav -- input.dsf output.wav
//! ```
//!
//! The DSF reader here handles the common case only (no metadata, no ID3 tag). DSF stores
//! each channel in blocks of 4096 bytes, least significant bit first, which maps directly onto
//! [`Decimator::process_planar`] and [`BitOrder::LsbFirst`].

use std::error::Error;
use std::fs;
use std::io::{BufWriter, Write};

use dsd_pcm::{BitOrder, Decimator};

/// The fields of a DSF header this example needs.
struct DsfFormat {
    channels: usize,
    dsd_rate: u32,
    lsb_first: bool,
    /// DSD bits per channel.
    samples: u64,
    /// Bytes per channel in each block.
    block_size: usize,
    /// Offset of the sample data in the file.
    data_start: usize,
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32, Box<dyn Error>> {
    let field = bytes.get(at..at + 4).ok_or("file too short")?;
    Ok(u32::from_le_bytes(field.try_into()?))
}

fn u64_at(bytes: &[u8], at: usize) -> Result<u64, Box<dyn Error>> {
    let field = bytes.get(at..at + 8).ok_or("file too short")?;
    Ok(u64::from_le_bytes(field.try_into()?))
}

/// Reads the `DSD `, `fmt ` and `data` chunk headers, which DSF puts at fixed offsets.
fn parse_header(file: &[u8]) -> Result<DsfFormat, Box<dyn Error>> {
    if file.get(0..4) != Some(b"DSD ") || file.get(28..32) != Some(b"fmt ") {
        return Err("not a DSF file".into());
    }
    let data_start = 28 + usize::try_from(u64_at(file, 32)?)?;
    if file.get(data_start..data_start + 4) != Some(b"data") {
        return Err("no data chunk after the format chunk".into());
    }
    Ok(DsfFormat {
        channels: usize::try_from(u32_at(file, 52)?)?,
        dsd_rate: u32_at(file, 56)?,
        lsb_first: u32_at(file, 60)? == 1,
        samples: u64_at(file, 64)?,
        block_size: usize::try_from(u32_at(file, 72)?)?,
        data_start: data_start + 12,
    })
}

/// Writes a WAV header for 32-bit float samples.
fn write_wav_header(
    out: &mut impl Write,
    rate: u32,
    channels: u16,
    frames: u64,
) -> Result<(), Box<dyn Error>> {
    let data_bytes = u32::try_from(frames * u64::from(channels) * 4)?;
    out.write_all(b"RIFF")?;
    out.write_all(&(36 + data_bytes).to_le_bytes())?;
    out.write_all(b"WAVEfmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&3u16.to_le_bytes())?; // IEEE float
    out.write_all(&channels.to_le_bytes())?;
    out.write_all(&rate.to_le_bytes())?;
    out.write_all(&(rate * u32::from(channels) * 4).to_le_bytes())?;
    out.write_all(&(channels * 4).to_le_bytes())?;
    out.write_all(&32u16.to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&data_bytes.to_le_bytes())?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output] = args.as_slice() else {
        return Err("usage: dsf_to_wav <input.dsf> <output.wav>".into());
    };
    let file = fs::read(input)?;
    let format = parse_header(&file)?;
    let order = if format.lsb_first {
        BitOrder::LsbFirst
    } else {
        BitOrder::MsbFirst
    };
    let mut decimator = Decimator::new(format.dsd_rate, format.channels)?.with_bit_order(order);

    // The last block is padded; only `samples / 8` bytes per channel are audio.
    let mut remaining = usize::try_from(format.samples / 8)?;
    let mut pcm = Vec::new();
    let group = format.block_size * format.channels;
    for blocks in file[format.data_start..].chunks_exact(group) {
        let used = remaining.min(format.block_size);
        let channels: Vec<&[u8]> = blocks
            .chunks_exact(format.block_size)
            .map(|block| &block[..used])
            .collect();
        decimator.process_planar(&channels, &mut pcm)?;
        remaining -= used;
        if remaining == 0 {
            break;
        }
    }
    decimator.finish(&mut pcm);

    let mut out = BufWriter::new(fs::File::create(output)?);
    let frames = (pcm.len() / format.channels) as u64;
    write_wav_header(
        &mut out,
        decimator.output_rate(),
        u16::try_from(format.channels)?,
        frames,
    )?;
    for sample in pcm {
        // The output is at most about ±0.7 for legal SACD levels: no clipping to handle.
        out.write_all(&(sample as f32).to_le_bytes())?;
    }
    out.flush()?;
    println!(
        "{} Hz DSD, {} channels -> {} Hz PCM, {frames} frames",
        format.dsd_rate,
        format.channels,
        decimator.output_rate()
    );
    Ok(())
}
