use core::mem;

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
pub mod util;

pub(crate) use imp::Group;

// Constant for h2 function that grabing the top 7 bits of the hash.
const MIN_HASH_LEN: usize = if mem::size_of::<usize>() < mem::size_of::<u64>() {
    mem::size_of::<usize>()
} else {
    mem::size_of::<u64>()
};

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
