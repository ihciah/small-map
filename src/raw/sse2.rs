#[cfg(target_arch = "x86")]
use core::arch::x86;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86;
use core::{mem, num::NonZeroU16};

use super::bitmask::BitMask;

pub(crate) type BitMaskWord = u16;
pub(crate) type NonZeroBitMaskWord = NonZeroU16;
pub(crate) const BITMASK_STRIDE: usize = 1;
pub(crate) const BITMASK_ITER_MASK: BitMaskWord = !0;

/// Abstraction over a group of control bytes which can be scanned in
/// parallel.
///
/// This implementation uses a 128-bit SSE value.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct Group(x86::__m128i);

// FIXME: https://github.com/rust-lang/rust-clippy/issues/3859
#[allow(clippy::use_self)]
impl Group {
    /// Number of bytes in the group.
    pub(crate) const WIDTH: usize = mem::size_of::<Self>();
    pub(crate) const LOWEST_MASK: [BitMaskWord; Group::WIDTH] = [
        0b0000_0000_0000_0000,
        0b0000_0000_0000_0001,
        0b0000_0000_0000_0011,
        0b0000_0000_0000_0111,
        0b0000_0000_0000_1111,
        0b0000_0000_0001_1111,
        0b0000_0000_0011_1111,
        0b0000_0000_0111_1111,
        0b0000_0000_1111_1111,
        0b0000_0001_1111_1111,
        0b0000_0011_1111_1111,
        0b0000_0111_1111_1111,
        0b0000_1111_1111_1111,
        0b0001_1111_1111_1111,
        0b0011_1111_1111_1111,
        0b0111_1111_1111_1111,
    ];

    /// Loads a group of bytes starting at the given address.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)] // unaligned load
    pub(crate) unsafe fn load(ptr: *const u8) -> Self {
        Group(x86::_mm_loadu_si128(ptr.cast()))
    }

    /// Returns a `BitMask` indicating all bytes in the group which have
    /// the given value.
    #[inline]
    pub(crate) fn match_byte(self, byte: u8) -> BitMask {
        #[allow(
            clippy::cast_possible_wrap, // byte: u8 as i8
            // byte: i32 as u16
            //   note: _mm_movemask_epi8 returns a 16-bit mask in a i32, the
            //   upper 16-bits of the i32 are zeroed:
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        unsafe {
            let cmp = x86::_mm_cmpeq_epi8(self.0, x86::_mm_set1_epi8(byte as i8));
            BitMask(x86::_mm_movemask_epi8(cmp) as u16)
        }
    }
}
