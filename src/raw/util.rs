// Branch prediction hint. This is currently only available on nightly but it
// consistently improves performance by 10-15%.
#[cfg(not(feature = "nightly"))]
pub(crate) use core::convert::identity as likely;
#[cfg(not(feature = "nightly"))]
pub(crate) use core::convert::identity as unlikely;
#[cfg(feature = "nightly")]
pub(crate) use core::intrinsics::{likely, unlikely};
// Use strict provenance functions if available.
#[cfg(feature = "nightly")]
use core::ptr::invalid_mut;
use core::{
    hash::{BuildHasher, Hash},
    mem,
    ptr::NonNull,
};

use crate::Equivalent;
// Implement it with a cast otherwise.
#[cfg(not(feature = "nightly"))]
#[inline(always)]
fn invalid_mut<T>(addr: usize) -> *mut T {
    addr as *mut T
}

/// Ensures that a single closure type across uses of this which, in turn prevents multiple
/// instances of any functions like RawTable::reserve from being generated
#[inline]
pub(crate) fn equivalent_key<Q, K, V>(k: &Q) -> impl Fn(&(K, V)) -> bool + '_
where
    Q: ?Sized + Equivalent<K>,
{
    move |x| k.equivalent(&x.0)
}

#[cfg(not(feature = "nightly"))]
#[inline]
pub(crate) fn make_hash<Q, S>(hash_builder: &S, val: &Q) -> u64
where
    Q: Hash + ?Sized,
    S: BuildHasher,
{
    use core::hash::Hasher;
    let mut state = hash_builder.build_hasher();
    val.hash(&mut state);
    state.finish()
}

#[cfg(feature = "nightly")]
#[inline]
pub(crate) fn make_hash<Q, S>(hash_builder: &S, val: &Q) -> u64
where
    Q: Hash + ?Sized,
    S: BuildHasher,
{
    hash_builder.hash_one(val)
}

pub(crate) trait SizedTypeProperties: Sized {
    const IS_ZERO_SIZED: bool = mem::size_of::<Self>() == 0;
    const NEEDS_DROP: bool = mem::needs_drop::<Self>();
}

impl<T> SizedTypeProperties for T {}

/// A reference to an empty bucket into which an can be inserted.
pub(crate) struct InsertSlot {
    pub(crate) index: usize,
}

/// A reference to a hash table bucket containing a `T`.
pub(crate) struct Bucket<T> {
    ptr: NonNull<T>,
}

// This Send impl is needed for rayon support. This is safe since Bucket is
// never exposed in a public API.
unsafe impl<T> Send for Bucket<T> {}

impl<T> Clone for Bucket<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<T> Bucket<T> {
    /// Creates a [`Bucket`] that contain pointer to the data.
    /// The pointer calculation is performed by calculating the
    /// offset from given `base` pointer (convenience for
    /// `base.as_ptr().sub(index)`).
    #[inline]
    pub(crate) unsafe fn from_base_index(base: NonNull<T>, index: usize) -> Self {
        // If mem::size_of::<T>() != 0 then return a pointer to an `element` in
        // the data part of the table (we start counting from "0", so that
        // in the expression T[last], the "last" index actually one less than the
        // "buckets" number in the table, i.e. "last = RawTableInner.bucket_mask"):
        let ptr = if T::IS_ZERO_SIZED {
            // won't overflow because index must be less than length (bucket_mask)
            // and bucket_mask is guaranteed to be less than `isize::MAX`
            // (see TableLayout::calculate_layout_for method)
            invalid_mut(index + 1)
        } else {
            base.as_ptr().add(index)
        };
        Self {
            ptr: NonNull::new_unchecked(ptr),
        }
    }

    /// Calculates the index of a [`Bucket`] as distance between two pointers
    /// (convenience for `base.as_ptr().offset_from(self.ptr.as_ptr()) as usize`).
    /// The returned value is in units of T: the distance in bytes divided by
    /// [`core::mem::size_of::<T>()`].
    #[inline]
    pub(crate) unsafe fn to_base_index(&self, base: NonNull<T>) -> usize {
        if T::IS_ZERO_SIZED {
            // this can not be UB
            self.ptr.as_ptr() as usize - 1
        } else {
            self.ptr.as_ptr().offset_from(base.as_ptr()) as usize
        }
    }

    /// Acquires the underlying raw pointer `*mut T` to `data`.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut T {
        if T::IS_ZERO_SIZED {
            // Just return an arbitrary ZST pointer which is properly aligned
            // invalid pointer is good enough for ZST
            invalid_mut(mem::align_of::<T>())
        } else {
            self.ptr.as_ptr()
        }
    }

    /// Executes the destructor (if any) of the pointed-to `data`.
    #[inline]
    pub(crate) unsafe fn drop(&self) {
        self.as_ptr().drop_in_place();
    }

    /// Reads the `value` from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    #[inline]
    pub(crate) unsafe fn read(&self) -> T {
        self.as_ptr().read()
    }

    /// Overwrites a memory location with the given `value` without reading
    /// or dropping the old value (like [`ptr::write`] function).
    #[inline]
    pub(crate) unsafe fn write(&self, val: T) {
        self.as_ptr().write(val);
    }

    /// Returns a shared immutable reference to the `value`.
    #[inline]
    pub(crate) unsafe fn as_ref<'a>(&self) -> &'a T {
        &*self.as_ptr()
    }

    /// Returns a unique mutable reference to the `value`.
    #[inline]
    pub(crate) unsafe fn as_mut<'a>(&self) -> &'a mut T {
        &mut *self.as_ptr()
    }

    /// Create a new [`Bucket`] that is offset from the `self` by the given
    /// `offset`. The pointer calculation is performed by calculating the
    /// offset from `self` pointer (convenience for `self.ptr.as_ptr().sub(offset)`).
    /// This function is used for iterators.
    #[inline]
    pub(crate) unsafe fn next_n(&self, offset: usize) -> Self {
        let ptr = if T::IS_ZERO_SIZED {
            // invalid pointer is good enough for ZST
            invalid_mut(self.ptr.as_ptr() as usize)
        } else {
            self.ptr.as_ptr().add(offset)
        };
        Self {
            ptr: NonNull::new_unchecked(ptr),
        }
    }
}
