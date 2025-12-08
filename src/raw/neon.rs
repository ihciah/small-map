use core::{arch::aarch64 as neon, mem, num::NonZeroU64};

use super::bitmask::BitMask;

pub(crate) type BitMaskWord = u64;
pub(crate) type NonZeroBitMaskWord = NonZeroU64;
pub(crate) const BITMASK_STRIDE: usize = 8;
pub(crate) const BITMASK_ITER_MASK: BitMaskWord = 0x8080_8080_8080_8080;

/// Abstraction over a group of control bytes which can be scanned in
/// parallel.
///
/// This implementation uses a 64-bit NEON value.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct Group(neon::uint8x8_t);

#[allow(clippy::use_self)]
impl Group {
    /// Number of bytes in the group.
    pub(crate) const WIDTH: usize = mem::size_of::<Self>();
    pub(crate) const LOWEST_MASK: [BitMaskWord; Group::WIDTH] = [
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0080,
        0x0000_0000_0000_8080,
        0x0000_0000_0080_8080,
        0x0000_0000_8080_8080,
        0x0000_0080_8080_8080,
        0x0000_8080_8080_8080,
        0x0080_8080_8080_8080,
    ];

    /// Loads a group of bytes starting at the given address.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)] // unaligned load
    pub(crate) unsafe fn load(ptr: *const u8) -> Self {
        Group(neon::vld1_u8(ptr))
    }

    /// Returns a `BitMask` indicating all bytes in the group which *may*
    /// have the given value.
    #[inline]
    pub(crate) fn match_byte(self, byte: u8) -> BitMask {
        unsafe {
            let cmp = neon::vceq_u8(self.0, neon::vdup_n_u8(byte));
            BitMask(neon::vget_lane_u64(neon::vreinterpret_u64_u8(cmp), 0))
        }
    }
}
