use core::{iter::FusedIterator, mem::MaybeUninit, ptr::NonNull};

use super::RawIterInner;
use crate::{inline::AlignedGroups, raw::util::Bucket};

pub(crate) struct RawIter<'a, const N: usize, T> {
    pub(crate) inner: RawIterInner<T>,
    pub(crate) aligned_groups: &'a AlignedGroups<N>,
    pub(crate) data: &'a [MaybeUninit<T>; N],
}

impl<'a, const N: usize, T> Iterator for RawIter<'a, N, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        match self.inner.next(self.aligned_groups.as_ptr(), unsafe {
            Bucket::from_base_index(NonNull::new_unchecked(self.data.as_ptr() as _), 0)
        }) {
            Some(x) => unsafe {
                let r = x.as_ref();
                Some(r)
            },
            None => None,
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize, T> ExactSizeIterator for RawIter<'_, N, T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<const N: usize, T> FusedIterator for RawIter<'_, N, T> {}

pub(crate) struct RawIntoIter<const N: usize, T> {
    pub(crate) inner: RawIterInner<T>,
    pub(crate) aligned_groups: AlignedGroups<N>,
    pub(crate) data: [MaybeUninit<T>; N],
}

impl<const N: usize, T> Iterator for RawIntoIter<N, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            Some(
                self.inner
                    .next(
                        self.aligned_groups.as_ptr(),
                        Bucket::from_base_index(NonNull::new_unchecked(self.data.as_ptr() as _), 0),
                    )?
                    .read(),
            )
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize, T> ExactSizeIterator for RawIntoIter<N, T> {}
impl<const N: usize, T> FusedIterator for RawIntoIter<N, T> {}

impl<const N: usize, T> Drop for RawIntoIter<N, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // Drop all remaining elements
            self.inner.drop_elements(
                self.aligned_groups.as_ptr(),
                Bucket::from_base_index(NonNull::new_unchecked(self.data.as_ptr() as _), 0),
            );
        }
    }
}
