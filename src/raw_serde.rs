// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the vega-prover project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/vega-prover

//! Raw, zstd-framed serde for large vectors of plain data.
//!
//! The prover's reusable state (the R1CS matrices, the cached `Az`/`Bz`/`Cz`
//! products, the witness) is dominated by multi-million-element `Vec<F>` and
//! `Vec<usize>`. Round-tripping those through the derived `serde`
//! implementations is the single largest cost of loading a precomputed prover:
//! every field element is visited as 32 individual bytes and then pushed
//! through a canonical-range check and a Montgomery multiplication.
//!
//! This module serializes such vectors as their raw little-endian memory,
//! split into independently compressed frames:
//!
//! ```text
//! u64 element_count
//! u64 frame_count
//! frame_count x (u64 raw_len, u64 zstd_len)
//! frames...
//! ```
//!
//! The whole thing is handed to the `Serializer` as a single `serialize_bytes`
//! payload, so any self-describing or compact format can carry it.
//!
//! Deserialization allocates the destination `Vec<T>` once, splits its byte
//! view at the frame boundaries and decompresses each frame straight into its
//! final position, in parallel. There is no per-element work and no large
//! intermediate buffer: peak memory is the result plus the compressed input.
//!
//! # Trust model
//!
//! The bytes are *not* validated beyond their lengths. They encode the
//! in-memory representation of `T` directly, which for a field element means
//! Montgomery-form limbs rather than a canonical integer. This is sound only
//! for data the caller already trusts (a prover artifact it wrote itself) and
//! only for types that implement the [`RawLayout`] marker.
//!
//! # Portability
//!
//! The encoding is little-endian host memory. Big-endian targets are rejected
//! at compile time, and [`RawLayout`] implementations are guarded by `const`
//! assertions on size and alignment, so a type whose layout does not match is
//! a compile error rather than a silent fallback to the slow path.

#![allow(unsafe_code)]

use std::mem::{align_of, size_of};

use rayon::prelude::*;
use serde::{
  Deserializer, Serializer,
  de::{Error as DeError, Visitor},
  ser::Error as SerError,
};

#[cfg(not(target_endian = "little"))]
compile_error!(
  "vega-prover's raw_serde encoding is little-endian host memory; \
   big-endian targets are not supported"
);

/// zstd level used for the frames. Higher levels barely shrink Montgomery
/// limbs and make saving noticeably slower.
const ZSTD_LEVEL: i32 = 1;

/// Target uncompressed size of one frame. Frames are compressed and
/// decompressed in parallel, so this trades per-frame zstd efficiency against
/// the number of available work items.
const FRAME_TARGET_BYTES: usize = 4 << 20;

/// Upper bound on the uncompressed size of a single frame, so that a damaged
/// header cannot drive an unbounded allocation even though the input is
/// trusted.
const MAX_FRAME_RAW_LEN: u64 = 1 << 30;

/// Size of the fixed part of the payload header: `element_count` and
/// `frame_count`.
const HEADER_LEN: usize = 16;

/// Size of one frame's entry in the table of contents: `raw_len`, `zstd_len`.
const TOC_ENTRY_LEN: usize = 16;

/// Marker for types whose in-memory representation may be copied verbatim to
/// and from a byte buffer.
///
/// # Safety
///
/// Implementing this trait asserts all of the following:
///
/// - `T` has no padding bytes, no pointers, no interior mutability and no
///   niches: *every* bit pattern of `size_of::<T>()` bytes is a valid, fully
///   initialized value of `T`.
/// - `T` carries no invariant that the rest of the code relies on beyond what
///   its bits encode, so a value reconstructed from bytes written by another
///   process of the same build behaves identically.
///
/// Field elements qualify because their representation is a plain array of
/// Montgomery limbs; this is exactly the assumption that
/// [`MontgomeryLimbs`](crate::traits::MontgomeryLimbs) already encodes, so
/// implementing that trait through `impl_montgomery_limbs!` also implements
/// this one.
pub unsafe trait RawLayout: Copy + Send + Sync {
  /// Compile-time guard, evaluated by [`check_layout`].
  ///
  /// It fails to compile for any type whose size is not a non-zero multiple of
  /// 8 bytes or whose alignment exceeds `u64`'s, because such a type would
  /// make the frame boundaries or the byte-slice view invalid.
  const LAYOUT_OK: () = {
    assert!(
      size_of::<Self>() > 0,
      "raw_serde cannot encode zero-sized types"
    );
    assert!(
      size_of::<Self>().is_multiple_of(8),
      "raw_serde requires a size that is a multiple of 8 bytes"
    );
    assert!(
      align_of::<Self>() <= align_of::<u64>(),
      "raw_serde requires an alignment of at most 8 bytes"
    );
  };
}

/// Forces evaluation of [`RawLayout::LAYOUT_OK`] for `T`.
#[inline]
const fn check_layout<T: RawLayout>() {
  #[allow(clippy::let_unit_value)]
  let () = T::LAYOUT_OK;
}

/// Implements [`RawLayout`] for one or more plain-data types.
///
/// # Example
/// ```ignore
/// crate::impl_raw_layout!(u64, usize);
/// ```
#[macro_export]
macro_rules! impl_raw_layout {
  ($($t:ty),+ $(,)?) => {
    $(
      #[allow(unsafe_code)]
      // SAFETY: see the `RawLayout` contract. The type is plain data with no
      // padding, no pointers and no invalid bit patterns, and the `const`
      // assertions in `RawLayout::LAYOUT_OK` reject an unsuitable layout.
      unsafe impl $crate::raw_serde::RawLayout for $t {}
    )+
  };
}

impl_raw_layout!(u64, i64, usize, isize);

/// Number of elements per frame for `T`, at least one.
#[inline]
fn frame_elems<T: RawLayout>() -> usize {
  (FRAME_TARGET_BYTES / size_of::<T>()).max(1)
}

/// Reinterprets a slice of plain data as its bytes.
#[inline]
fn as_bytes<T: RawLayout>(v: &[T]) -> &[u8] {
  // SAFETY: `RawLayout` guarantees `T` is plain data without padding, so the
  // `len * size_of::<T>()` bytes behind `v` are initialized and readable, and
  // `u8` has no alignment requirement.
  unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Serializes `v` as the framed, zstd-compressed raw bytes documented above.
///
/// Intended to be used through `#[serde(with = "crate::raw_serde")]`.
pub fn serialize<T, S>(v: &[T], serializer: S) -> Result<S::Ok, S::Error>
where
  T: RawLayout,
  S: Serializer,
{
  check_layout::<T>();

  let bytes = as_bytes(v);
  let chunk = frame_elems::<T>() * size_of::<T>();
  let frames = bytes
    .par_chunks(chunk)
    .map(|frame| zstd::bulk::compress(frame, ZSTD_LEVEL).map(|z| (frame.len(), z)))
    .collect::<Result<Vec<_>, _>>()
    .map_err(S::Error::custom)?;

  let payload_len =
    HEADER_LEN + TOC_ENTRY_LEN * frames.len() + frames.iter().map(|(_, z)| z.len()).sum::<usize>();
  let mut payload = Vec::with_capacity(payload_len);
  payload.extend_from_slice(&(v.len() as u64).to_le_bytes());
  payload.extend_from_slice(&(frames.len() as u64).to_le_bytes());
  for (raw_len, z) in &frames {
    payload.extend_from_slice(&(*raw_len as u64).to_le_bytes());
    payload.extend_from_slice(&(z.len() as u64).to_le_bytes());
  }
  for (_, z) in &frames {
    payload.extend_from_slice(z);
  }

  serializer.serialize_bytes(&payload)
}

/// A frame's position in the payload and in the destination vector.
struct Frame<'a> {
  compressed: &'a [u8],
  raw_len: usize,
}

/// Parses the payload header and table of contents.
fn parse_frames<E: DeError>(payload: &[u8]) -> Result<(usize, Vec<Frame<'_>>), E> {
  let u64_at = |at: usize| u64::from_le_bytes(payload[at..at + 8].try_into().unwrap());

  if payload.len() < HEADER_LEN {
    return Err(E::custom("raw_serde: payload shorter than its header"));
  }
  let elements = u64_at(0);
  let frame_count = u64_at(8) as usize;

  let toc_end = HEADER_LEN.saturating_add(TOC_ENTRY_LEN.saturating_mul(frame_count));
  if payload.len() < toc_end {
    return Err(E::custom("raw_serde: truncated frame table"));
  }

  let mut at = toc_end;
  let mut frames = Vec::with_capacity(frame_count);
  for i in 0..frame_count {
    let raw_len = u64_at(HEADER_LEN + TOC_ENTRY_LEN * i);
    let zstd_len = u64_at(HEADER_LEN + TOC_ENTRY_LEN * i + 8);
    if raw_len > MAX_FRAME_RAW_LEN {
      return Err(E::custom("raw_serde: implausible frame length"));
    }
    let end = at
      .checked_add(zstd_len as usize)
      .filter(|end| *end <= payload.len())
      .ok_or_else(|| E::custom("raw_serde: truncated frame"))?;
    frames.push(Frame {
      compressed: &payload[at..end],
      raw_len: raw_len as usize,
    });
    at = end;
  }
  if at != payload.len() {
    return Err(E::custom("raw_serde: trailing bytes after the last frame"));
  }

  Ok((elements as usize, frames))
}

/// Decompresses `frames` into a freshly allocated `Vec<T>` of `elements`
/// elements, one frame at a time, in parallel and without a staging buffer.
fn decode_into_vec<T: RawLayout, E: DeError>(
  elements: usize,
  frames: &[Frame<'_>],
) -> Result<Vec<T>, E> {
  let total_raw: usize = frames.iter().map(|f| f.raw_len).sum();
  let expected = elements
    .checked_mul(size_of::<T>())
    .ok_or_else(|| E::custom("raw_serde: element count overflows"))?;
  if total_raw != expected {
    return Err(E::custom(format!(
      "raw_serde: frames hold {total_raw} bytes but {elements} elements need {expected}"
    )));
  }

  let mut out: Vec<T> = Vec::with_capacity(elements);
  // SAFETY: `out` has capacity for `elements` values of `T`, so the pointer is
  // valid for `expected` bytes. The slice is only ever written to (by
  // `decompress_to_buffer`) before `set_len` publishes it, and `u8` imposes no
  // alignment constraint. `RawLayout` guarantees that any byte pattern written
  // here is a valid `T`.
  let dst: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), expected) };

  // Split the destination into one disjoint piece per frame.
  let mut pieces: Vec<&mut [u8]> = Vec::with_capacity(frames.len());
  let mut rest = dst;
  for frame in frames {
    let (piece, tail) = rest.split_at_mut(frame.raw_len);
    pieces.push(piece);
    rest = tail;
  }
  debug_assert!(rest.is_empty());

  pieces
    .into_par_iter()
    .zip(frames.par_iter())
    .try_for_each(|(piece, frame)| -> Result<(), String> {
      let written = zstd::bulk::decompress_to_buffer(frame.compressed, piece)
        .map_err(|e| format!("raw_serde: {e}"))?;
      if written != frame.raw_len {
        return Err(format!(
          "raw_serde: frame decompressed to {written} bytes, expected {}",
          frame.raw_len
        ));
      }
      Ok(())
    })
    .map_err(E::custom)?;

  // SAFETY: every byte of the first `elements` values was written above, and
  // `RawLayout` makes any bit pattern a valid `T`.
  unsafe { out.set_len(elements) };
  Ok(out)
}

/// Deserializes a vector written by [`serialize`].
///
/// Intended to be used through `#[serde(with = "crate::raw_serde")]`.
pub fn deserialize<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
  T: RawLayout,
  D: Deserializer<'de>,
{
  check_layout::<T>();

  struct RawVisitor<T>(std::marker::PhantomData<T>);

  impl<'de, T: RawLayout> Visitor<'de> for RawVisitor<T> {
    type Value = Vec<T>;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      write!(f, "a zstd-framed raw byte payload")
    }

    fn visit_bytes<E: DeError>(self, payload: &[u8]) -> Result<Self::Value, E> {
      let (elements, frames) = parse_frames::<E>(payload)?;
      decode_into_vec::<T, E>(elements, &frames)
    }

    fn visit_borrowed_bytes<E: DeError>(self, payload: &'de [u8]) -> Result<Self::Value, E> {
      self.visit_bytes(payload)
    }

    fn visit_byte_buf<E: DeError>(self, payload: Vec<u8>) -> Result<Self::Value, E> {
      self.visit_bytes(&payload)
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
      // Formats without a native byte-string type (e.g. JSON) hand the payload
      // over as a sequence of integers.
      let mut payload = Vec::with_capacity(seq.size_hint().unwrap_or(0));
      while let Some(b) = seq.next_element::<u8>()? {
        payload.push(b);
      }
      self.visit_bytes(&payload)
    }
  }

  deserializer.deserialize_bytes(RawVisitor::<T>(std::marker::PhantomData))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::MontgomeryLimbs;
  use serde::{Deserialize, Serialize};

  #[derive(Serialize, Deserialize, PartialEq, Debug)]
  #[serde(bound = "T: RawLayout")]
  struct Wrapper<T: RawLayout> {
    #[serde(with = "crate::raw_serde")]
    v: Vec<T>,
    tail: u32,
  }

  fn round_trip<T: RawLayout + PartialEq + std::fmt::Debug>(v: Vec<T>) {
    let w = Wrapper { v, tail: 0xDEAD };
    let bytes = bincode::serialize(&w).unwrap();
    let back: Wrapper<T> = bincode::deserialize(&bytes).unwrap();
    assert_eq!(w, back);
  }

  fn scalars<F: ff::PrimeField + MontgomeryLimbs + RawLayout>(n: usize) -> Vec<F> {
    // A mix of zero, small and full-width values, which is what the prover's
    // vectors actually hold.
    (0..n)
      .map(|i| match i % 4 {
        0 => F::ZERO,
        1 => F::ONE,
        2 => -F::ONE,
        _ => F::from(i as u64) * (-F::ONE),
      })
      .collect()
  }

  fn exercise_sizes<F: ff::PrimeField + MontgomeryLimbs + RawLayout>() {
    let per_frame = frame_elems::<F>();
    for n in [
      0,
      1,
      2,
      per_frame - 1,
      per_frame,
      per_frame + 1,
      3 * per_frame,
    ] {
      round_trip(scalars::<F>(n));
    }
  }

  #[test]
  fn round_trip_usize() {
    let per_frame = frame_elems::<usize>();
    for n in [0usize, 1, 7, per_frame, per_frame + 3, 2 * per_frame + 1] {
      round_trip((0..n).collect::<Vec<usize>>());
    }
  }

  #[test]
  fn round_trip_u64() {
    round_trip(
      (0u64..100_000)
        .map(|i| i.wrapping_mul(0x0101_0101_0101_0101))
        .collect::<Vec<u64>>(),
    );
  }

  #[test]
  fn round_trip_bn254_scalar() {
    exercise_sizes::<crate::provider::bn254::types::Scalar>();
  }

  #[test]
  fn round_trip_t256_scalar() {
    exercise_sizes::<halo2curves::t256::Fq>();
  }

  #[test]
  fn round_trip_p256_scalar() {
    exercise_sizes::<halo2curves::secp256r1::Fq>();
  }

  #[test]
  fn montgomery_limbs_are_preserved() {
    type F = crate::provider::bn254::types::Scalar;
    let v = scalars::<F>(1000);
    let limbs: Vec<[u64; 4]> = v.iter().map(|f| *f.to_limbs()).collect();
    let w = Wrapper { v, tail: 1 };
    let back: Wrapper<F> = bincode::deserialize(&bincode::serialize(&w).unwrap()).unwrap();
    let back_limbs: Vec<[u64; 4]> = back.v.iter().map(|f| *f.to_limbs()).collect();
    assert_eq!(limbs, back_limbs);
  }

  #[test]
  fn rejects_a_truncated_payload() {
    let w = Wrapper {
      v: (0u64..10_000).collect::<Vec<u64>>(),
      tail: 0,
    };
    let bytes = bincode::serialize(&w).unwrap();
    // Drop the trailing `tail` plus a slice of the last frame; the payload
    // length prefix then promises more bytes than are present.
    let truncated = &bytes[..bytes.len() - 64];
    assert!(bincode::deserialize::<Wrapper<u64>>(truncated).is_err());
  }

  #[test]
  fn rejects_a_corrupt_frame_table() {
    let w = Wrapper {
      v: (0u64..10_000).collect::<Vec<u64>>(),
      tail: 0,
    };
    let mut bytes = bincode::serialize(&w).unwrap();
    // The payload starts after bincode's 8-byte length prefix: bump the
    // element count so it no longer matches the frames.
    let at = 8;
    let bumped = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) + 1;
    bytes[at..at + 8].copy_from_slice(&bumped.to_le_bytes());
    assert!(bincode::deserialize::<Wrapper<u64>>(&bytes).is_err());
  }

  #[test]
  fn frames_are_actually_split() {
    let per_frame = frame_elems::<u64>();
    let w = Wrapper {
      v: (0u64..(3 * per_frame as u64)).collect::<Vec<u64>>(),
      tail: 0,
    };
    let bytes = bincode::serialize(&w).unwrap();
    let frame_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    assert_eq!(frame_count, 3);
  }
}
