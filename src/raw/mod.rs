use core::{marker::PhantomData, mem, ptr::NonNull};

use crate::raw::util::{unlikely, SizedTypeProperties};

cfg_if! {
    // Use the SSE2 implementation if possible: it allows us to scan 16 buckets
    // at once instead of 8. We don't bother with AVX since it would require
    // runtime dispatch and wouldn't gain us much anyways: the probability of
    // finding a match drops off drastically after the first few buckets.
    //
    // I attempted an implementation on ARM using NEON instructions, but it
    // turns out that most NEON instructions have multi-cycle latency, which in
    // the end outweighs any gains over the generic implementation.
    if #[cfg(all(
        target_feature = "sse2",
        any(target_arch = "x86", target_arch = "x86_64"),
    ))] {
        mod sse2;
        use sse2 as imp;
    } else if #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        // NEON intrinsics are currently broken on big-endian targets.
        // See https://github.com/rust-lang/stdarch/issues/1484.
        target_endian = "little",
    ))] {
        mod neon;
        use neon as imp;
    } else {
        mod generic;
        use generic as imp;
    }
}

mod bitmask;
pub mod iter;
pub mod util;

pub(crate) use imp::{BitMaskWord, Group};

use self::{
    bitmask::{BitMask, BitMaskIter},
    util::Bucket,
};

// Constant for h2 function that grabing the top 7 bits of the hash.
const MIN_HASH_LEN: usize = if mem::size_of::<usize>() < mem::size_of::<u64>() {
    mem::size_of::<usize>()
} else {
    mem::size_of::<u64>()
};

/// Control byte value for an empty bucket.
pub(crate) const EMPTY: u8 = 0b1111_1111;
/// Control byte value for a deleted bucket.
pub(crate) const DELETED: u8 = 0b1000_0000;

/// Secondary hash function, saved in the low 7 bits of the control byte.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn h2(hash: u64) -> u8 {
    // Grab the top 7 bits of the hash. While the hash is normally a full 64-bit
    // value, some hash functions (such as FxHash) produce a usize result
    // instead, which means that the top 32 bits are 0 on 32-bit platforms.
    // So we use MIN_HASH_LEN constant to handle this.
    let top7 = hash >> (MIN_HASH_LEN * 8 - 7);
    (top7 & 0x7f) as u8 // truncation
}

pub struct RawIterInner<T> {
    len: usize,
    current_group: BitMaskIter,
    group_offset: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T> Send for RawIterInner<T> {}
unsafe impl<T> Sync for RawIterInner<T> {}

impl<T> RawIterInner<T> {
    #[inline]
    pub(crate) unsafe fn new(init_group: BitMask, len: usize) -> Self {
        Self {
            len,
            current_group: init_group.into_iter(),
            group_offset: 0,
            _marker: PhantomData,
        }
    }

    #[inline]
    unsafe fn next_impl(&mut self, group_base: NonNull<u8>, base: Bucket<T>) -> Bucket<T> {
        loop {
            if let Some(index) = self.current_group.next() {
                return unsafe { base.next_n(index + self.group_offset) };
            }

            self.group_offset += Group::WIDTH;
            self.current_group = unsafe { Group::load_aligned(group_base.as_ptr().add(self.group_offset)) }
                .match_full()
                .into_iter();
        }
    }

    #[inline]
    pub(crate) fn next(&mut self, group_base: NonNull<u8>, base: Bucket<T>) -> Option<Bucket<T>> {
        if unlikely(self.len == 0) {
            return None;
        }
        self.len -= 1;
        Some(unsafe { self.next_impl(group_base, base) })
    }

    #[inline]
    pub(crate) fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl<T> RawIterInner<T> {
    pub(crate) unsafe fn drop_elements(&mut self, group_base: NonNull<u8>, base: Bucket<T>) {
        if T::NEEDS_DROP && self.len != 0 {
            while let Some(item) = self.next(group_base, base.clone()) {
                unsafe { item.drop(); }
            }
        }
    }
}
