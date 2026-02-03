// Branch prediction hint. This is currently only available on nightly but it
// consistently improves performance by 10-15%.
#[cfg(not(feature = "nightly"))]
pub(crate) use core::convert::identity as likely;
#[cfg(not(feature = "nightly"))]
pub(crate) use core::convert::identity as unlikely;
#[cfg(feature = "nightly")]
pub(crate) use core::intrinsics::{likely, unlikely};
use core::hash::{BuildHasher, Hash};

use crate::Equivalent;

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
    #[allow(clippy::manual_hash_one)]
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
    const NEEDS_DROP: bool = core::mem::needs_drop::<Self>();
}

impl<T> SizedTypeProperties for T {}

/// A reference to an empty bucket into which an can be inserted.
pub(crate) struct InsertSlot {
    pub(crate) index: usize,
}
