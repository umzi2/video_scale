//! Loading the ONNX graph with an opset ONNX Runtime accepts.
//!
//! ONNX Runtime 1.31 rejects models stamped with an unreleased opset (28 here).
//! Every operator in this model has a stable schema far below opset 26, so the
//! `ai.onnx` version is rewritten in memory before the graph is loaded.

use anyhow::{Context, Result, bail, ensure};
use std::{fs, ops::Range, path::Path};

/// Version written into every newer `ai.onnx` opset import.
const OPSET: u64 = 26;

/// Reads `model`, lowering an unreleased `ai.onnx` opset if the file has one.
pub fn model_bytes(model: &Path) -> Result<Vec<u8>> {
    let mut bytes = fs::read(model).with_context(|| format!("reading {}", model.display()))?;
    let patched = patch_model(&mut bytes)?;
    ensure!(
        patched == 1,
        "expected one ai.onnx opset import, patched {patched}"
    );
    Ok(bytes)
}

/// Rewrites every `ModelProto.opset_import`; returns how many changed.
fn patch_model(bytes: &mut [u8]) -> Result<usize> {
    let mut patched = 0;
    for field in fields(bytes)? {
        if field.number != 8 {
            continue;
        }
        if let Value::Bytes(range) = field.value
            && patch_opset(&mut bytes[range])?
        {
            patched += 1;
        }
    }
    Ok(patched)
}

/// Rewrites `OperatorSetIdProto.version` for `ai.onnx` when it is newer than [`OPSET`].
fn patch_opset(entry: &mut [u8]) -> Result<bool> {
    let mut domain: Option<Range<usize>> = None;
    let mut version = None;
    for field in fields(entry)? {
        match (field.number, field.value) {
            (1, Value::Bytes(range)) => domain = Some(range),
            (2, Value::Varint { value, span }) => version = Some((value, span)),
            _ => {}
        }
    }
    let Some((value, span)) = version else {
        return Ok(false);
    };
    let is_onnx = match domain {
        None => true,
        Some(range) => {
            let domain = &entry[range];
            domain.is_empty() || domain == b"ai.onnx"
        }
    };
    if !is_onnx || value <= OPSET {
        return Ok(false);
    }
    ensure!(
        span.len() == 1,
        "unexpected opset version encoding: {value}"
    );
    entry[span.start] = OPSET as u8;
    Ok(true)
}

/// One protobuf field, with value positions instead of borrows so callers can
/// rewrite the buffer while they walk it.
struct Field {
    number: u64,
    value: Value,
}

enum Value {
    Varint { value: u64, span: Range<usize> },
    Bytes(Range<usize>),
    Fixed,
}

fn fields(bytes: &[u8]) -> Result<Vec<Field>> {
    let mut parsed = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let (key, next) = varint(bytes, pos)?;
        let mut field = Field {
            number: key >> 3,
            value: Value::Fixed,
        };
        pos = match key & 7 {
            0 => {
                let (value, end) = varint(bytes, next)?;
                field.value = Value::Varint {
                    value,
                    span: next..end,
                };
                end
            }
            1 => next + 8,
            2 => {
                let (length, start) = varint(bytes, next)?;
                let end = start + length as usize;
                ensure!(end <= bytes.len(), "truncated protobuf message");
                field.value = Value::Bytes(start..end);
                end
            }
            5 => next + 4,
            wire => bail!("unsupported protobuf wire type {wire}"),
        };
        ensure!(pos <= bytes.len(), "truncated protobuf field");
        parsed.push(field);
    }
    Ok(parsed)
}

fn varint(bytes: &[u8], mut pos: usize) -> Result<(u64, usize)> {
    let mut value = 0;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(pos).context("truncated protobuf varint")?;
        value |= u64::from(byte & 0x7f) << shift;
        pos += 1;
        if byte & 0x80 == 0 {
            return Ok((value, pos));
        }
    }
    bail!("invalid protobuf varint");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_only_the_onnx_opset() {
        // ModelProto { opset_import: {domain: "ai.onnx", version: 28},
        //             opset_import: {domain: "custom", version: 7} }
        let mut model = vec![0x42, 11, 0x0a, 7];
        model.extend(b"ai.onnx");
        model.extend([0x10, 28]);
        model.extend([0x42, 10, 0x0a, 6]);
        model.extend(b"custom");
        model.extend([0x10, 7]);

        assert_eq!(patch_model(&mut model).unwrap(), 1);
        assert!(model.windows(2).any(|w| w == [0x10, OPSET as u8]));
        assert!(model.windows(2).any(|w| w == [0x10, 7]));
    }

    #[test]
    fn rejects_truncated_messages() {
        let mut model = vec![0x42, 11, 0x0a, 7];
        model.extend(b"ai");
        assert!(patch_model(&mut model).is_err());
    }
}
