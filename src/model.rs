//! Loading the ONNX graph with the opset ONNX Runtime can accept.

use anyhow::{Context, Result, bail, ensure};
use std::{fs, path::Path};

/// ONNX Runtime 1.31 rejects models stamped with an unreleased opset (28 here).
/// Every operator in this model has a stable schema far below opset 26, so the
/// `ai.onnx` version is rewritten in memory before loading.
pub fn model_bytes(model: &Path) -> Result<Vec<u8>> {
    let mut bytes = fs::read(model).with_context(|| format!("reading {}", model.display()))?;
    let mut pos = 0;
    let mut patched = 0;
    while pos < bytes.len() {
        let (key, next) = varint(&bytes, pos)?;
        let (field, wire) = (key >> 3, key & 7);
        match wire {
            0 => pos = varint(&bytes, next)?.1,
            1 => pos = next + 8,
            5 => pos = next + 4,
            2 => {
                let (len, start) = varint(&bytes, next)?;
                let end = start + len as usize;
                ensure!(end <= bytes.len(), "truncated ONNX model");
                if field == 8 && patch_opset(&mut bytes[start..end])? {
                    patched += 1;
                }
                pos = end;
            }
            _ => bail!("unsupported ONNX field {field}"),
        }
    }
    ensure!(
        patched == 1,
        "expected one ai.onnx opset import, patched {patched}"
    );
    Ok(bytes)
}

fn varint(bytes: &[u8], mut pos: usize) -> Result<(u64, usize)> {
    let mut value = 0;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(pos).context("truncated ONNX varint")?;
        value |= u64::from(byte & 0x7f) << shift;
        pos += 1;
        if byte & 0x80 == 0 {
            return Ok((value, pos));
        }
    }
    bail!("invalid ONNX varint");
}

/// Rewrite `OperatorSetIdProto`'s version for `ai.onnx` if it is newer than 26.
fn patch_opset(entry: &mut [u8]) -> Result<bool> {
    let (mut domain, mut version) = (None, None);
    let mut pos = 0;
    while pos < entry.len() {
        let (key, next) = varint(entry, pos)?;
        let (field, wire) = (key >> 3, key & 7);
        match (field, wire) {
            (1, 2) => {
                let (len, start) = varint(entry, next)?;
                domain = Some(&entry[start..start + len as usize]);
                pos = start + len as usize;
            }
            (2, 0) => {
                let (value, end) = varint(entry, next)?;
                version = Some((value, next, end));
                pos = end;
            }
            (_, 0) => pos = varint(entry, next)?.1,
            (_, 2) => {
                let (len, start) = varint(entry, next)?;
                pos = start + len as usize;
            }
            (_, 5) => pos = next + 4,
            _ => bail!("unsupported ONNX opset field {field}"),
        }
    }
    let Some((value, start, end)) = version else {
        return Ok(false);
    };
    let is_onnx = domain.is_none_or(|d| d.is_empty() || d == b"ai.onnx");
    if !is_onnx || value <= 26 {
        return Ok(false);
    }
    ensure!(
        end - start == 1,
        "unexpected opset version encoding: {value}"
    );
    entry[start] = 26;
    Ok(true)
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
        let mut patched = 0;
        let mut pos = 0;
        while pos < model.len() {
            let (key, next) = varint(&model, pos).unwrap();
            let (len, start) = varint(&model, next).unwrap();
            let end = start + len as usize;
            if key >> 3 == 8 && patch_opset(&mut model[start..end]).unwrap() {
                patched += 1;
            }
            pos = end;
        }
        assert_eq!(patched, 1);
        assert!(model.windows(2).any(|w| w == [0x10, 26]));
        assert!(model.windows(2).any(|w| w == [0x10, 7]));
    }
}
