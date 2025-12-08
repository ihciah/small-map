#![allow(clippy::manual_map)]
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(feature = "nightly", feature(core_intrinsics))]

//! A small inline SIMD-accelerated hashmap designed for small amounts of data.
//!
//! `SmallMap` stores data inline when the number of elements is small (up to N),
//! and automatically spills to the heap when it grows beyond that threshold.
//!
//! # Examples
//!
//! ```
//! use small_map::SmallMap;
//!
//! let mut map: SmallMap<8, &str, i32> = SmallMap::new();
//! map.insert("a", 1);
//! map.insert("b", 2);
//!
//! assert_eq!(map.get(&"a"), Some(&1));
//! assert!(map.contains_key(&"b"));
//! assert!(map.is_inline()); // Still stored inline
//! ```

use core::{
    hash::{BuildHasher, Hash},
    hint::unreachable_unchecked,
    iter::FusedIterator,
    ops::Index,
};
use std::{
    collections::hash_map::RandomState,
    fmt::{self, Debug},
};

use hashbrown::HashMap;
use inline::Inline;

#[macro_use]
mod macros;
mod inline;
mod raw;

pub use hashbrown::Equivalent;

#[cfg(feature = "serde")]
mod serde;

#[cfg(feature = "fxhash")]
pub type FxSmallMap<const N: usize, K, V> =
    SmallMap<N, K, V, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
#[cfg(feature = "ahash")]
pub type ASmallMap<const N: usize, K, V> =
    SmallMap<N, K, V, core::hash::BuildHasherDefault<ahash::AHasher>>;

#[cfg(feature = "fxhash")]
type DefaultInlineHasher = core::hash::BuildHasherDefault<rustc_hash::FxHasher>;
#[cfg(all(not(feature = "fxhash"), feature = "ahash"))]
type DefaultInlineHasher = core::hash::BuildHasherDefault<ahash::AHasher>;
#[cfg(all(not(feature = "fxhash"), not(feature = "ahash")))]
type DefaultInlineHasher = RandomState;

#[derive(Clone)]
pub enum SmallMap<const N: usize, K, V, S = RandomState, SI = DefaultInlineHasher> {
    Heap(HashMap<K, V, S>),
    Inline(Inline<N, K, V, S, SI>),
}

impl<const N: usize, K, V, S, SI> Debug for SmallMap<N, K, V, S, SI>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<const N: usize, K, V, S: Default, SI: Default> Default for SmallMap<N, K, V, S, SI> {
    /// Creates an empty `SmallMap<N, K, V, S, SI>`, with the `Default` value for the hasher.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    ///
    /// use small_map::SmallMap;
    ///
    /// // You can specify all types of SmallMap, including N and hasher.
    /// // Created map is empty and don't allocate memory
    /// let map: SmallMap<8, u32, String> = SmallMap::default();
    /// assert_eq!(map.capacity(), 8);
    /// let map: SmallMap<8, u32, String, RandomState> = SmallMap::default();
    /// assert_eq!(map.capacity(), 8);
    /// ```
    #[inline]
    fn default() -> Self {
        Self::with_hashers(Default::default(), Default::default())
    }
}

impl<const N: usize, K, V, S: Default, SI: Default> SmallMap<N, K, V, S, SI> {
    /// Creates an empty `SmallMap`.
    #[inline]
    pub fn new() -> Self {
        Self::with_hashers(Default::default(), Default::default())
    }

    /// Creates an empty `SmallMap` with the specified capacity.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is smaller than N, the hash map will not allocate.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity > N {
            Self::Heap(HashMap::with_capacity_and_hasher(
                capacity,
                Default::default(),
            ))
        } else {
            Self::Inline(Inline::new(Default::default(), Default::default()))
        }
    }
}

impl<const N: usize, K, V, S, SI> SmallMap<N, K, V, S, SI> {
    /// Creates an empty `SmallMap` which will use the given hash builder to hash
    /// keys. It will be allocated with the given allocator.
    ///
    /// The hash map is initially created with a capacity of N, so it will not allocate until it
    /// its size bigger than inline size N.
    /// # Examples
    ///
    /// ```
    /// use core::hash::BuildHasherDefault;
    ///
    /// use small_map::SmallMap;
    ///
    /// let s = BuildHasherDefault::<ahash::AHasher>::default();
    /// let mut map = SmallMap::<8, _, _, _>::with_hasher(s);
    /// map.insert(1, 2);
    /// ```
    #[inline]
    pub fn with_hasher(hash_builder: S) -> Self
    where
        SI: Default,
    {
        Self::Inline(Inline::new(SI::default(), hash_builder))
    }

    /// Creates an empty `SmallMap` which will use the given hash builders to hash
    /// keys. It will be allocated with the given allocator.
    /// # Examples
    /// ```
    /// use std::collections::hash_map::RandomState;
    ///
    /// use small_map::SmallMap;
    /// let heap_hasher = RandomState::new();
    /// let inline_hasher = RandomState::new();
    /// let mut map = SmallMap::<8, _, _, _, _>::with_hashers(heap_hasher, inline_hasher);
    /// map.insert(1, 2);
    /// ```
    #[inline]
    pub const fn with_hashers(heap_hasher: S, inline_hasher: SI) -> Self {
        Self::Inline(Inline::new(inline_hasher, heap_hasher))
    }

    /// Creates an empty `SmallMap` with the specified capacity, using `heap_hasher`
    /// to hash the keys.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is smaller than or eq to N, the hash map will not allocate.
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, heap_hasher: S) -> Self
    where
        SI: Default,
    {
        if capacity > N {
            Self::Heap(HashMap::with_capacity_and_hasher(capacity, heap_hasher))
        } else {
            Self::Inline(Inline::new(SI::default(), heap_hasher))
        }
    }

    /// Creates an empty `SmallMap` with the specified capacity, using `heap_hasher`
    /// to hash the keys for heap storage, and `inline_hasher` for inline storage.
    ///
    /// # Examples
    /// ```
    /// use std::collections::hash_map::RandomState;
    ///
    /// use small_map::SmallMap;
    /// let heap_hasher = RandomState::new();
    /// let inline_hasher = RandomState::new();
    /// let mut map =
    ///     SmallMap::<8, _, _, _, _>::with_capacity_and_hashers(16, heap_hasher, inline_hasher);
    /// map.insert(1, 2);
    /// ```
    #[inline]
    pub fn with_capacity_and_hashers(capacity: usize, heap_hasher: S, inline_hasher: SI) -> Self {
        if capacity > N {
            Self::Heap(HashMap::with_capacity_and_hasher(capacity, heap_hasher))
        } else {
            Self::Inline(Inline::new(inline_hasher, heap_hasher))
        }
    }

    /// What branch it is now.
    #[inline]
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline(..))
    }

    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// This number is a lower bound; the `SmallMap<N, K, V>` might be able to hold
    /// more, but is guaranteed to be able to hold at least this many.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let map: SmallMap<8, i32, i32> = SmallMap::with_capacity(100);
    /// assert_eq!(map.len(), 0);
    /// assert!(map.capacity() >= 100);
    ///
    /// let map: SmallMap<8, i32, i32> = SmallMap::with_capacity(2);
    /// assert_eq!(map.len(), 0);
    /// assert!(map.capacity() >= 8);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        match self {
            SmallMap::Heap(inner) => inner.capacity(),
            SmallMap::Inline(_) => N,
        }
    }
}

impl<const N: usize, K, V, S, SI> SmallMap<N, K, V, S, SI>
where
    K: Eq + Hash,
    S: BuildHasher,
    SI: BuildHasher,
{
    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, &str> = SmallMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[inline]
    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        match self {
            SmallMap::Heap(inner) => inner.get(k),
            SmallMap::Inline(inner) => inner.get(k),
        }
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, i32> = SmallMap::new();
    /// map.insert(1, 10);
    /// if let Some(v) = map.get_mut(&1) {
    ///     *v = 20;
    /// }
    /// assert_eq!(map.get(&1), Some(&20));
    /// ```
    #[inline]
    pub fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        match self {
            SmallMap::Heap(inner) => inner.get_mut(k),
            SmallMap::Inline(inner) => inner.get_mut(k),
        }
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, &str> = SmallMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get_key_value(&1), Some((&1, &"a")));
    /// assert_eq!(map.get_key_value(&2), None);
    /// ```
    #[inline]
    pub fn get_key_value<Q>(&self, k: &Q) -> Option<(&K, &V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        match self {
            SmallMap::Heap(inner) => inner.get_key_value(k),
            SmallMap::Inline(inner) => inner.get_key_value(k),
        }
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, &str> = SmallMap::new();
    /// map.insert(1, "a");
    /// assert!(map.contains_key(&1));
    /// assert!(!map.contains_key(&2));
    /// ```
    #[inline]
    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.get(k).is_some()
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, `None` is returned.
    /// If the map did have this key present, the value is updated, and the old value is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, &str> = SmallMap::new();
    /// assert_eq!(map.insert(1, "a"), None);
    /// assert_eq!(map.insert(1, "b"), Some("a"));
    /// assert_eq!(map.get(&1), Some(&"b"));
    /// ```
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self {
            SmallMap::Heap(inner) => inner.insert(key, value),
            SmallMap::Inline(inner) => {
                if raw::util::likely(!inner.is_full()) {
                    return inner.insert(key, value);
                }
                let heap =
                    HashMap::with_capacity_and_hasher(N * 2, unsafe { inner.take_heap_hasher() });

                let heap = match core::mem::replace(self, SmallMap::Heap(heap)) {
                    SmallMap::Heap(_) => unsafe { unreachable_unchecked() },
                    SmallMap::Inline(inline) => {
                        let heap = match self {
                            SmallMap::Heap(heap) => heap,
                            SmallMap::Inline(_) => unsafe { unreachable_unchecked() },
                        };
                        for (k, v) in inline.into_iter() {
                            // SAFETY: We're moving elements from inline storage to heap.
                            // All keys are unique because they came from a valid map.
                            unsafe { heap.insert_unique_unchecked(k, v) };
                        }
                        heap
                    }
                };
                heap.insert(key, value)
            }
        }
    }

    /// Removes a key from the map, returning the value at the key if the key was previously in the
    /// map.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, &str> = SmallMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[inline]
    pub fn remove<Q>(&mut self, k: &Q) -> Option<V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.remove_entry(k) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    /// Removes a key from the map, returning the stored key and value if the key was previously in
    /// the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, &str> = SmallMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove_entry(&1), Some((1, "a")));
    /// assert_eq!(map.remove_entry(&1), None);
    /// ```
    #[inline]
    pub fn remove_entry<Q>(&mut self, k: &Q) -> Option<(K, V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        match self {
            SmallMap::Heap(inner) => inner.remove_entry(k),
            SmallMap::Inline(inner) => inner.remove_entry(k),
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` for which `f(&k, &mut v)` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, i32> = SmallMap::new();
    /// for i in 0..8 {
    ///     map.insert(i, i * 10);
    /// }
    /// map.retain(|&k, _| k % 2 == 0);
    /// assert_eq!(map.len(), 4);
    /// ```
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        match self {
            SmallMap::Heap(inner) => inner.retain(f),
            SmallMap::Inline(inner) => inner.retain(&mut f),
        }
    }
}

impl<const N: usize, K, V, S, SI> SmallMap<N, K, V, S, SI> {
    /// Clears the map.
    ///
    /// This method clears the map and keeps the allocated memory for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, ()> = SmallMap::new();
    /// for i in 0..16 {
    ///     map.insert(i, ());
    /// }
    /// assert!(!map.is_inline());
    /// assert_eq!(map.len(), 16);
    ///
    /// map.clear();
    /// assert!(!map.is_inline());
    /// assert_eq!(map.len(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        match self {
            SmallMap::Heap(inner) => {
                inner.clear();
            }
            // Safety: We're about to destroy this inner, so it doesn't need its hasher
            SmallMap::Inline(inner) => unsafe {
                *self = Self::Inline(Inline::new(
                    inner.take_inline_hasher(),
                    inner.take_heap_hasher(),
                ))
            },
        }
    }
}

pub enum Iter<'a, const N: usize, K, V> {
    Heap(hashbrown::hash_map::Iter<'a, K, V>),
    Inline(inline::Iter<'a, N, K, V>),
}

impl<'a, const N: usize, K, V> Iterator for Iter<'a, N, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        match self {
            Iter::Heap(inner) => inner.next(),
            Iter::Inline(inner) => inner.next(),
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Iter::Heap(inner) => inner.size_hint(),
            Iter::Inline(inner) => inner.size_hint(),
        }
    }
}

impl<const N: usize, K, V> ExactSizeIterator for Iter<'_, N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Iter::Heap(inner) => inner.len(),
            Iter::Inline(inner) => inner.len(),
        }
    }
}

impl<const N: usize, K, V> FusedIterator for Iter<'_, N, K, V> {}

/// An iterator over the keys of a `SmallMap`.
pub struct Keys<'a, const N: usize, K, V> {
    inner: Iter<'a, N, K, V>,
}

impl<'a, const N: usize, K, V> Iterator for Keys<'a, N, K, V> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<&'a K> {
        match self.inner.next() {
            Some((k, _)) => Some(k),
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize, K, V> ExactSizeIterator for Keys<'_, N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<const N: usize, K, V> FusedIterator for Keys<'_, N, K, V> {}

/// An iterator over the values of a `SmallMap`.
pub struct Values<'a, const N: usize, K, V> {
    inner: Iter<'a, N, K, V>,
}

impl<'a, const N: usize, K, V> Iterator for Values<'a, N, K, V> {
    type Item = &'a V;

    #[inline]
    fn next(&mut self) -> Option<&'a V> {
        match self.inner.next() {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize, K, V> ExactSizeIterator for Values<'_, N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<const N: usize, K, V> FusedIterator for Values<'_, N, K, V> {}

/// A consuming iterator over the keys of a `SmallMap`.
pub struct IntoKeys<const N: usize, K, V> {
    inner: IntoIter<N, K, V>,
}

impl<const N: usize, K, V> Iterator for IntoKeys<N, K, V> {
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<K> {
        match self.inner.next() {
            Some((k, _)) => Some(k),
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize, K, V> ExactSizeIterator for IntoKeys<N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<const N: usize, K, V> FusedIterator for IntoKeys<N, K, V> {}

/// A consuming iterator over the values of a `SmallMap`.
pub struct IntoValues<const N: usize, K, V> {
    inner: IntoIter<N, K, V>,
}

impl<const N: usize, K, V> Iterator for IntoValues<N, K, V> {
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<V> {
        match self.inner.next() {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize, K, V> ExactSizeIterator for IntoValues<N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<const N: usize, K, V> FusedIterator for IntoValues<N, K, V> {}

pub enum IntoIter<const N: usize, K, V> {
    Heap(hashbrown::hash_map::IntoIter<K, V>),
    Inline(inline::IntoIter<N, K, V>),
}

impl<const N: usize, K, V> Iterator for IntoIter<N, K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        match self {
            IntoIter::Heap(inner) => inner.next(),
            IntoIter::Inline(inner) => inner.next(),
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IntoIter::Heap(inner) => inner.size_hint(),
            IntoIter::Inline(inner) => inner.size_hint(),
        }
    }
}
impl<const N: usize, K, V> ExactSizeIterator for IntoIter<N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            IntoIter::Heap(inner) => inner.len(),
            IntoIter::Inline(inner) => inner.len(),
        }
    }
}

impl<const N: usize, K, V> FusedIterator for IntoIter<N, K, V> {}

impl<const N: usize, K, V, S, SI> IntoIterator for SmallMap<N, K, V, S, SI> {
    type Item = (K, V);
    type IntoIter = IntoIter<N, K, V>;

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    #[inline]
    fn into_iter(self) -> IntoIter<N, K, V> {
        match self {
            SmallMap::Heap(inner) => IntoIter::Heap(inner.into_iter()),
            SmallMap::Inline(inner) => IntoIter::Inline(inner.into_iter()),
        }
    }
}

impl<'a, const N: usize, K, V, S, SI> IntoIterator for &'a SmallMap<N, K, V, S, SI> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, N, K, V>;

    /// Creates an iterator over the entries of a `HashMap` in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    #[inline]
    fn into_iter(self) -> Iter<'a, N, K, V> {
        match self {
            SmallMap::Heap(inner) => Iter::Heap(inner.iter()),
            SmallMap::Inline(inner) => Iter::Inline(inner.iter()),
        }
    }
}

impl<const N: usize, K, V, S, SI> SmallMap<N, K, V, S, SI> {
    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, i32> = SmallMap::new();
    /// assert_eq!(map.len(), 0);
    /// map.insert(1, 10);
    /// assert_eq!(map.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            SmallMap::Heap(inner) => inner.len(),
            SmallMap::Inline(inner) => inner.len(),
        }
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, i32, i32> = SmallMap::new();
    /// assert!(map.is_empty());
    /// map.insert(1, 10);
    /// assert!(!map.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            SmallMap::Heap(inner) => inner.is_empty(),
            SmallMap::Inline(inner) => inner.is_empty(),
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, &str, i32> = SmallMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {key} val: {val}");
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, N, K, V> {
        match self {
            SmallMap::Heap(inner) => Iter::Heap(inner.iter()),
            SmallMap::Inline(inner) => Iter::Inline(inner.iter()),
        }
    }

    /// An iterator visiting all keys in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, &str, i32> = SmallMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    ///
    /// for key in map.keys() {
    ///     println!("{key}");
    /// }
    /// ```
    #[inline]
    pub fn keys(&self) -> Keys<'_, N, K, V> {
        Keys { inner: self.iter() }
    }

    /// An iterator visiting all values in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, &str, i32> = SmallMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    ///
    /// for val in map.values() {
    ///     println!("{val}");
    /// }
    /// ```
    #[inline]
    pub fn values(&self) -> Values<'_, N, K, V> {
        Values { inner: self.iter() }
    }

    /// Creates a consuming iterator visiting all the keys in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, &str, i32> = SmallMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    ///
    /// let keys: Vec<_> = map.into_keys().collect();
    /// ```
    #[inline]
    pub fn into_keys(self) -> IntoKeys<N, K, V> {
        IntoKeys {
            inner: self.into_iter(),
        }
    }

    /// Creates a consuming iterator visiting all the values in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallMap;
    ///
    /// let mut map: SmallMap<8, &str, i32> = SmallMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    ///
    /// let values: Vec<_> = map.into_values().collect();
    /// ```
    #[inline]
    pub fn into_values(self) -> IntoValues<N, K, V> {
        IntoValues {
            inner: self.into_iter(),
        }
    }
}

// PartialEq implementation
impl<const N: usize, K, V, S, SI> PartialEq for SmallMap<N, K, V, S, SI>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
    SI: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter()
            .all(|(key, value)| other.get(key) == Some(value))
    }
}

impl<const N: usize, K, V, S, SI> Eq for SmallMap<N, K, V, S, SI>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
    SI: BuildHasher,
{
}

// Index implementation
impl<const N: usize, K, V, S, SI, Q> Index<&Q> for SmallMap<N, K, V, S, SI>
where
    K: Eq + Hash,
    Q: ?Sized + Hash + Equivalent<K>,
    S: BuildHasher,
    SI: BuildHasher,
{
    type Output = V;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the `SmallMap`.
    #[inline]
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

// Extend implementation
impl<const N: usize, K, V, S, SI> Extend<(K, V)> for SmallMap<N, K, V, S, SI>
where
    K: Eq + Hash,
    S: BuildHasher,
    SI: BuildHasher,
{
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

// FromIterator implementation
impl<const N: usize, K, V, S, SI> FromIterator<(K, V)> for SmallMap<N, K, V, S, SI>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
    SI: BuildHasher + Default,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = SmallMap::new();
        map.extend(iter);
        map
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::hash_map::RandomState, rc::Rc};

    use hashbrown::HashMap;
    use rand::Rng;

    use crate::SmallMap;

    #[test]
    fn basic_op() {
        let mut map = SmallMap::<16, String, String>::default();
        map.insert("hello".to_string(), "world".to_string());
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("hello").unwrap(), "world");
        map.insert("hello2".to_string(), "world2".to_string());
        assert_eq!(map.get("hello2").unwrap(), "world2");
        assert_eq!(map.len(), 2);

        assert_eq!(
            map.remove_entry("hello").unwrap(),
            ("hello".to_string(), "world".to_string())
        );
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("hello2").unwrap(), "world2");
        assert_eq!(map.remove("hello2").unwrap(), "world2".to_string());
        assert_eq!(map.len(), 0);
        assert!(map.get("hello").is_none());

        map.insert("hello3".to_string(), "world3".to_string());
        map.insert("hello4".to_string(), "world4".to_string());
        assert_eq!(map.len(), 2);
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.get("hello3").is_none());
        assert!(map.get("hello4").is_none());
    }

    #[test]
    fn multi_group_with_padding() {
        let mut map = SmallMap::<33, i32, i32>::default();
        for i in 0..33 {
            map.insert(i, i * 2);
        }
        for i in 0..33 {
            assert_eq!(*map.get(&i).unwrap(), i * 2);
        }
        assert!(map.is_inline());

        for i in 33..64 {
            map.insert(i, i * 2);
        }
        assert!(!map.is_inline());
        for i in 0..64 {
            assert_eq!(*map.get(&i).unwrap(), i * 2);
        }
    }

    #[test]
    fn clear_keep_state() {
        let mut map = SmallMap::<16, i32, i32>::default();
        for i in 0..32 {
            map.insert(i, i * 2);
        }
        assert!(!map.is_inline());
        map.clear();
        assert!(!map.is_inline());
    }

    #[test]
    fn fuzzing() {
        let mut smallmap = SmallMap::<16, i32, i32>::default();
        let mut hashmap = HashMap::<i32, i32, RandomState>::default();
        for _ in 0..1000000 {
            let op = Operation::random();
            op.exec(&mut smallmap, &mut hashmap);
        }

        enum Operation {
            Insert(i32, i32),
            Remove(i32),
            Get(i32),
            ModifyIfExist(i32, i32),
        }
        impl Operation {
            fn random() -> Self {
                let mut rng = rand::rng();

                let choice: u8 = rng.random();
                match choice % 4 {
                    0 => Operation::Insert(rng.random_range(0..32), rng.random()),
                    1 => Operation::Remove(rng.random_range(0..32)),
                    2 => Operation::Get(rng.random_range(0..32)),
                    3 => Operation::ModifyIfExist(rng.random_range(0..32), rng.random()),
                    _ => unreachable!(),
                }
            }

            fn exec<const N: usize, S1: core::hash::BuildHasher, S2: core::hash::BuildHasher>(
                self,
                sm: &mut SmallMap<N, i32, i32, S1>,
                hm: &mut HashMap<i32, i32, S2>,
            ) {
                match self {
                    Operation::Insert(k, v) => {
                        if sm.len() == sm.capacity() {
                            return;
                        }
                        assert_eq!(sm.insert(k, v), hm.insert(k, v));
                        assert!(sm.is_inline());
                    }
                    Operation::Remove(k) => {
                        assert_eq!(sm.remove(&k), hm.remove(&k));
                    }
                    Operation::Get(k) => {
                        assert_eq!(sm.get(&k), hm.get(&k));
                    }
                    Operation::ModifyIfExist(k, nv) => {
                        let (sv, hv) = (sm.get_mut(&k), hm.get_mut(&k));
                        assert_eq!(sv, hv);
                        if let Some(v) = sv {
                            *v = nv;
                        }
                        if let Some(v) = hv {
                            *v = nv;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn drop_chk() {
        let (probe1, checker1) = drop_checker();
        let (probe2, checker2) = drop_checker();
        let (probe3, checker3) = drop_checker();
        let mut map = SmallMap::<8, _, _>::default();
        map.insert(1, probe1);
        map.insert(2, probe2);
        map.insert(3, probe3);
        assert_eq!(map.len(), 3);
        let mut it = map.into_iter();
        drop(it.next());
        drop(it);
        checker1.assert_drop();
        checker2.assert_drop();
        checker3.assert_drop();

        fn drop_checker() -> (DropProbe, DropChecker) {
            let flag = Rc::new(RefCell::new(false));
            (DropProbe { flag: flag.clone() }, DropChecker { flag })
        }

        struct DropChecker {
            flag: Rc<RefCell<bool>>,
        }

        impl DropChecker {
            fn assert_drop(self) {
                assert!(*self.flag.borrow())
            }
        }

        struct DropProbe {
            flag: Rc<RefCell<bool>>,
        }

        impl Drop for DropProbe {
            fn drop(&mut self) {
                *self.flag.borrow_mut() = true;
            }
        }
    }

    #[test]
    fn clone_after_delete() {
        let mut map: SmallMap<8, i32, String> = SmallMap::new();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        map.insert(3, "three".to_string());

        assert_eq!(map.len(), 3);
        assert!(map.is_inline());

        map.remove(&2);
        assert_eq!(map.len(), 2);

        let cloned = map.clone();

        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.get(&1), Some(&"one".to_string()));
        assert_eq!(cloned.get(&3), Some(&"three".to_string()));
        assert_eq!(cloned.get(&2), None);

        let mut count = 0;
        for _ in cloned.iter() {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn clone_after_multiple_deletes() {
        let mut map: SmallMap<16, i32, i32> = SmallMap::new();

        for i in 0..10 {
            map.insert(i, i * 100);
        }
        assert_eq!(map.len(), 10);

        for i in (0..10).step_by(2) {
            map.remove(&i);
        }
        assert_eq!(map.len(), 5);

        let cloned = map.clone();
        assert_eq!(cloned.len(), 5);

        for i in (1..10).step_by(2) {
            assert_eq!(cloned.get(&i), Some(&(i * 100)));
        }

        for i in (0..10).step_by(2) {
            assert_eq!(cloned.get(&i), None);
        }

        let collected: Vec<_> = cloned.iter().collect();
        assert_eq!(collected.len(), 5);
    }

    #[test]
    fn insert_into_cloned_after_delete() {
        let mut map: SmallMap<8, i32, String> = SmallMap::new();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        map.insert(3, "three".to_string());

        map.remove(&2);

        let mut cloned = map.clone();

        cloned.insert(4, "four".to_string());

        assert_eq!(cloned.len(), 3);
        assert_eq!(cloned.get(&1), Some(&"one".to_string()));
        assert_eq!(cloned.get(&3), Some(&"three".to_string()));
        assert_eq!(cloned.get(&4), Some(&"four".to_string()));
    }

    #[test]
    fn into_iter_cloned_after_delete() {
        let mut map: SmallMap<8, i32, String> = SmallMap::new();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        map.insert(3, "three".to_string());

        map.remove(&2);

        let cloned = map.clone();

        let items: Vec<_> = cloned.into_iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn clone_compaction() {
        // Test that clone compacts the data array
        let mut map: SmallMap<8, i32, i32> = SmallMap::new();

        // Fill with gaps: insert 0-7, then delete evens
        for i in 0..8 {
            map.insert(i, i * 10);
        }
        for i in (0..8).step_by(2) {
            map.remove(&i);
        }

        // Clone should compact: 4 elements at indices 0-3
        let cloned = map.clone();
        assert_eq!(cloned.len(), 4);

        // Verify all odd keys are present
        for i in (1..8).step_by(2) {
            assert_eq!(cloned.get(&i), Some(&(i * 10)));
        }

        // Insert into cloned should work correctly
        let mut cloned = cloned;
        cloned.insert(100, 1000);
        assert_eq!(cloned.len(), 5);
        assert_eq!(cloned.get(&100), Some(&1000));
    }

    #[test]
    fn clone_empty_after_delete_all() {
        let mut map: SmallMap<8, i32, i32> = SmallMap::new();

        map.insert(1, 10);
        map.insert(2, 20);
        map.remove(&1);
        map.remove(&2);

        assert_eq!(map.len(), 0);

        let cloned = map.clone();
        assert_eq!(cloned.len(), 0);
        assert!(cloned.is_empty());

        // Should be able to insert into cloned empty map
        let mut cloned = cloned;
        cloned.insert(3, 30);
        assert_eq!(cloned.get(&3), Some(&30));
    }

    #[test]
    fn contains_key_test() {
        let mut map: SmallMap<8, i32, &str> = SmallMap::new();
        map.insert(1, "a");
        map.insert(2, "b");

        assert!(map.contains_key(&1));
        assert!(map.contains_key(&2));
        assert!(!map.contains_key(&3));

        map.remove(&1);
        assert!(!map.contains_key(&1));
    }

    #[test]
    fn keys_values_test() {
        let mut map: SmallMap<8, i32, i32> = SmallMap::new();
        for i in 0..5 {
            map.insert(i, i * 10);
        }

        let keys: Vec<_> = map.keys().cloned().collect();
        assert_eq!(keys.len(), 5);
        for i in 0..5 {
            assert!(keys.contains(&i));
        }

        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 5);
        for i in 0..5 {
            assert!(values.contains(&(i * 10)));
        }
    }

    #[test]
    fn into_keys_values_test() {
        let mut map: SmallMap<8, i32, i32> = SmallMap::new();
        for i in 0..5 {
            map.insert(i, i * 10);
        }

        let keys: Vec<_> = map.clone().into_keys().collect();
        assert_eq!(keys.len(), 5);

        let values: Vec<_> = map.into_values().collect();
        assert_eq!(values.len(), 5);
    }

    #[test]
    fn retain_test() {
        let mut map: SmallMap<8, i32, i32> = SmallMap::new();
        for i in 0..8 {
            map.insert(i, i * 10);
        }

        map.retain(|&k, _| k % 2 == 0);
        assert_eq!(map.len(), 4);

        for i in (0..8).step_by(2) {
            assert!(map.contains_key(&i));
        }
        for i in (1..8).step_by(2) {
            assert!(!map.contains_key(&i));
        }
    }

    #[test]
    fn partial_eq_test() {
        let mut map1: SmallMap<8, i32, i32> = SmallMap::new();
        let mut map2: SmallMap<8, i32, i32> = SmallMap::new();

        map1.insert(1, 10);
        map1.insert(2, 20);

        map2.insert(2, 20);
        map2.insert(1, 10);

        assert_eq!(map1, map2);

        map2.insert(3, 30);
        assert_ne!(map1, map2);
    }

    #[test]
    fn index_test() {
        let mut map: SmallMap<8, i32, &str> = SmallMap::new();
        map.insert(1, "a");
        map.insert(2, "b");

        assert_eq!(map[&1], "a");
        assert_eq!(map[&2], "b");
    }

    #[test]
    #[should_panic(expected = "no entry found for key")]
    fn index_panic_test() {
        let map: SmallMap<8, i32, &str> = SmallMap::new();
        let _ = map[&1];
    }

    #[test]
    fn extend_test() {
        let mut map: SmallMap<8, i32, i32> = SmallMap::new();
        map.insert(1, 10);

        map.extend([(2, 20), (3, 30)]);
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.get(&3), Some(&30));
    }

    #[test]
    fn from_iterator_test() {
        let map: SmallMap<8, i32, i32> = [(1, 10), (2, 20), (3, 30)].into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.get(&3), Some(&30));
    }

    #[test]
    fn retain_heap_test() {
        let mut map: SmallMap<4, i32, i32> = SmallMap::new();
        for i in 0..10 {
            map.insert(i, i * 10);
        }
        assert!(!map.is_inline());

        map.retain(|&k, _| k % 2 == 0);
        assert_eq!(map.len(), 5);
    }
}
