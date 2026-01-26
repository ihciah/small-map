//! A small inline SIMD-accelerated hash set based on [`SmallMap`].

use core::{
    fmt::{self, Debug},
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
};
use std::collections::hash_map::RandomState;

use crate::{DefaultInlineHasher, Equivalent, SmallMap, DEFAULT_LINEAR_THRESHOLD};

/// A hash set implemented as a [`SmallMap`] where the value is `()`.
///
/// As with the [`SmallMap`] type, a `SmallSet` requires that the elements
/// implement the [`Eq`] and [`Hash`] traits. This can frequently be achieved by
/// using `#[derive(PartialEq, Eq, Hash)]`. If you implement these yourself,
/// it is important that the following property holds:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two elements are equal, their hashes must be equal.
///
/// It is a logic error for an item to be modified in such a way that the
/// item's hash, as determined by the [`Hash`] trait, or its equality, as
/// determined by the [`Eq`] trait, changes while it is in the set. This is
/// normally only possible through [`Cell`], [`RefCell`], global state, I/O, or
/// unsafe code.
///
/// It is also a logic error for the [`Hash`] implementation of an element to panic.
/// This is generally only possible if the trait is implemented manually. If a
/// panic does occur then the contents of the `SmallSet` may become corrupted and
/// some items may be dropped from the table.
///
/// # Type Parameters
///
/// - `N`: Maximum number of elements to store inline (must be > 0). When the set exceeds this size,
///   it automatically spills to a heap-allocated hash table.
///
/// - `T`: Element type. Must implement `Eq + Hash` for most operations.
///
/// - `SH`: Hasher for heap storage. Default: [`RandomState`].
///
/// - `SI`: Hasher for inline storage. Default: `DefaultInlineHasher`.
///
/// - `LINEAR_THRESHOLD`: Threshold for switching between linear and SIMD search. Default:
///   [`DEFAULT_LINEAR_THRESHOLD`].
///
/// # Examples
///
/// ```
/// use small_map::SmallSet;
///
/// // Type inference lets us omit an explicit type signature (which
/// // would be `SmallSet<8, String>` in this example).
/// let mut books = SmallSet::<8, _>::new();
///
/// // Add some books.
/// books.insert("A Dance With Dragons".to_string());
/// books.insert("To Kill a Mockingbird".to_string());
/// books.insert("The Odyssey".to_string());
/// books.insert("The Great Gatsby".to_string());
///
/// // Check for a specific one.
/// if !books.contains("The Winds of Winter") {
///     println!(
///         "We have {} books, but The Winds of Winter ain't one.",
///         books.len()
///     );
/// }
///
/// // Remove a book.
/// books.remove("The Odyssey");
///
/// // Iterate over everything.
/// for book in &books {
///     println!("{}", book);
/// }
/// ```
///
/// The easiest way to use `SmallSet` with a custom type is to derive
/// [`Eq`] and [`Hash`]. We must also derive [`PartialEq`]. This will in the
/// future be implied by [`Eq`].
///
/// ```
/// use small_map::SmallSet;
///
/// #[derive(Hash, Eq, PartialEq, Debug)]
/// struct Viking {
///     name: String,
///     power: usize,
/// }
///
/// let mut vikings = SmallSet::<8, _>::new();
///
/// vikings.insert(Viking {
///     name: "Einar".to_string(),
///     power: 9,
/// });
/// vikings.insert(Viking {
///     name: "Einar".to_string(),
///     power: 9,
/// });
/// vikings.insert(Viking {
///     name: "Olaf".to_string(),
///     power: 4,
/// });
/// vikings.insert(Viking {
///     name: "Harald".to_string(),
///     power: 8,
/// });
///
/// // Use derived implementation to print the vikings.
/// for x in &vikings {
///     println!("{:?}", x);
/// }
/// ```
///
/// A `SmallSet` with a fixed list of elements can be initialized from an array:
///
/// ```
/// use small_map::SmallSet;
///
/// let viking_names: SmallSet<8, &'static str> = ["Einar", "Olaf", "Harald"].into_iter().collect();
/// // use the values stored in the set
/// ```
///
/// [`Cell`]: core::cell::Cell
/// [`RefCell`]: core::cell::RefCell
#[derive(Clone)]
pub struct SmallSet<
    const N: usize,
    T,
    SH = RandomState,
    SI = DefaultInlineHasher,
    const LINEAR_THRESHOLD: usize = DEFAULT_LINEAR_THRESHOLD,
> {
    pub(crate) map: SmallMap<N, T, (), SH, SI, LINEAR_THRESHOLD>,
}

impl<const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize> Debug
    for SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<const N: usize, T, SH: Default, SI: Default, const LINEAR_THRESHOLD: usize> Default
    for SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
{
    /// Creates an empty `SmallSet<N, T, SH, SI>`, with the `Default` value for the hasher.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    ///
    /// use small_map::SmallSet;
    ///
    /// let set: SmallSet<8, u32> = SmallSet::default();
    /// assert_eq!(set.capacity(), 8);
    /// let set: SmallSet<8, u32, RandomState> = SmallSet::default();
    /// assert_eq!(set.capacity(), 8);
    /// ```
    #[inline]
    fn default() -> Self {
        Self::with_hashers(Default::default(), Default::default())
    }
}

impl<const N: usize, T, SH: Default, SI: Default, const LINEAR_THRESHOLD: usize>
    SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
{
    /// Creates an empty `SmallSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, &str> = SmallSet::new();
    /// set.insert("a");
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::with_hashers(Default::default(), Default::default())
    }

    /// Creates an empty `SmallSet` with the specified capacity.
    ///
    /// The set will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is smaller than N, the set will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, &str> = SmallSet::with_capacity(100);
    /// assert!(!set.is_inline());
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: SmallMap::with_capacity(capacity),
        }
    }
}

impl<const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize>
    SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
{
    /// Creates an empty `SmallSet` which will use the given hash builder to hash
    /// elements.
    ///
    /// The set is initially created with a capacity of N, so it will not allocate until it
    /// grows beyond that threshold.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    ///
    /// use small_map::SmallSet;
    ///
    /// let s = RandomState::default();
    /// let mut set = SmallSet::<8, _>::with_hasher(s);
    /// set.insert(1);
    /// ```
    #[inline]
    pub fn with_hasher(hash_builder: SH) -> Self
    where
        SI: Default,
    {
        Self {
            map: SmallMap::with_hasher(hash_builder),
        }
    }

    /// Creates an empty `SmallSet` which will use the given hash builders to hash
    /// elements.
    ///
    /// The set is initially created with a capacity of N, so it will not allocate until it
    /// grows beyond that threshold.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    ///
    /// use small_map::SmallSet;
    ///
    /// let heap_hasher = RandomState::default();
    /// let inline_hasher = rapidhash::fast::RandomState::default();
    /// let mut set = SmallSet::<8, _>::with_hashers(heap_hasher, inline_hasher);
    /// set.insert(1);
    /// ```
    #[inline]
    pub fn with_hashers(heap_hasher: SH, inline_hasher: SI) -> Self {
        Self {
            map: SmallMap::with_hashers(heap_hasher, inline_hasher),
        }
    }

    /// Creates an empty `SmallSet` with the specified capacity, using
    /// `heap_hasher` to hash the elements.
    ///
    /// The set will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is smaller than or eq to N, the set will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    ///
    /// use small_map::SmallSet;
    ///
    /// let s = RandomState::default();
    /// let mut set = SmallSet::<8, _>::with_capacity_and_hasher(10, s);
    /// set.insert(1);
    /// ```
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, heap_hasher: SH) -> Self
    where
        SI: Default,
    {
        Self {
            map: SmallMap::with_capacity_and_hasher(capacity, heap_hasher),
        }
    }

    /// Creates an empty `SmallSet` with the specified capacity, using
    /// `heap_hasher` for heap storage and `inline_hasher` for inline storage.
    ///
    /// The set will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is smaller than or eq to N, the set will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    ///
    /// use small_map::SmallSet;
    ///
    /// let heap_hasher = RandomState::default();
    /// let inline_hasher = rapidhash::fast::RandomState::default();
    /// let mut set = SmallSet::<8, _>::with_capacity_and_hashers(10, heap_hasher, inline_hasher);
    /// set.insert(1);
    /// ```
    #[inline]
    pub fn with_capacity_and_hashers(capacity: usize, heap_hasher: SH, inline_hasher: SI) -> Self {
        Self {
            map: SmallMap::with_capacity_and_hashers(capacity, heap_hasher, inline_hasher),
        }
    }

    /// Returns `true` if the set is using inline storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = SmallSet::new();
    /// assert!(set.is_inline());
    ///
    /// for i in 0..10 {
    ///     set.insert(i);
    /// }
    /// assert!(!set.is_inline());
    /// ```
    #[inline]
    pub fn is_inline(&self) -> bool {
        self.map.is_inline()
    }

    /// Returns the number of elements the set can hold without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let set: SmallSet<8, i32> = SmallSet::new();
    /// assert_eq!(set.capacity(), 8);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }
}

impl<const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize>
    SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
where
    T: Eq + Hash,
    SH: BuildHasher,
    SI: BuildHasher,
{
    /// Returns `true` if the set contains a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = SmallSet::new();
    /// set.insert(1);
    /// assert!(set.contains(&1));
    /// assert!(!set.contains(&2));
    /// ```
    #[inline]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.contains_key(value)
    }

    /// Returns a reference to the value in the set, if any, that is equal to the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let set: SmallSet<8, _> = [1, 2, 3].into_iter().collect();
    /// assert_eq!(set.get(&2), Some(&2));
    /// assert_eq!(set.get(&4), None);
    /// ```
    #[inline]
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.get_key_value(value).map(|(k, _)| k)
    }

    /// Adds a value to the set.
    ///
    /// If the set did not have this value present, `true` is returned.
    ///
    /// If the set did have this value present, `false` is returned, and the
    /// entry is not updated.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = SmallSet::new();
    ///
    /// assert_eq!(set.insert(2), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        self.map.insert(value, ()).is_none()
    }

    /// Adds a value to the set, replacing the existing value, if any, that is equal to the given
    /// one. Returns the replaced value.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = SmallSet::new();
    /// set.insert(1);
    ///
    /// assert_eq!(set.replace(1), Some(1));
    /// assert_eq!(set.replace(2), None);
    /// ```
    #[inline]
    pub fn replace(&mut self, value: T) -> Option<T> {
        // Try to remove the old value first
        let old_value = self.take(&value);
        // Insert the new value
        self.insert(value);
        old_value
    }

    /// Removes a value from the set. Returns `true` if the value was present in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = SmallSet::new();
    /// set.insert(1);
    /// assert_eq!(set.remove(&1), true);
    /// assert_eq!(set.remove(&1), false);
    /// ```
    #[inline]
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.remove(value).is_some()
    }

    /// Removes and returns the value in the set, if any, that is equal to the given one.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = SmallSet::new();
    /// set.insert(1);
    /// assert_eq!(set.take(&1), Some(1));
    /// assert_eq!(set.take(&1), None);
    /// ```
    #[inline]
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.remove_entry(value).map(|(k, _)| k)
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = (0..8).collect();
    /// set.retain(|&k| k % 2 == 0);
    /// assert_eq!(set.len(), 4);
    /// ```
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.map.retain(|k, _| f(k))
    }
}

impl<const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize>
    SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
{
    /// Clears the set, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = SmallSet::new();
    /// set.insert(1);
    /// set.clear();
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Returns the number of elements in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = SmallSet::new();
    /// assert_eq!(set.len(), 0);
    /// set.insert(1);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the set contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set: SmallSet<8, i32> = SmallSet::new();
    /// assert!(set.is_empty());
    /// set.insert(1);
    /// assert!(!set.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// An iterator visiting all elements in arbitrary order.
    /// The iterator element type is `&'a T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use small_map::SmallSet;
    ///
    /// let mut set = SmallSet::<8, _>::new();
    /// set.insert("a");
    /// set.insert("b");
    ///
    /// // Will print in an arbitrary order.
    /// for x in set.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, N, T> {
        Iter {
            inner: self.map.keys(),
        }
    }
}

// Iterator types
pub struct Iter<'a, const N: usize, T> {
    inner: crate::Keys<'a, N, T, ()>,
}

impl<'a, const N: usize, T> Clone for Iter<'a, N, T> {
    #[inline]
    fn clone(&self) -> Self {
        Iter {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, const N: usize, T> Iterator for Iter<'a, N, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize, T> ExactSizeIterator for Iter<'_, N, T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<const N: usize, T> FusedIterator for Iter<'_, N, T> {}

pub struct IntoIter<const N: usize, T> {
    inner: crate::IntoKeys<N, T, ()>,
}

impl<const N: usize, T> Iterator for IntoIter<N, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize, T> ExactSizeIterator for IntoIter<N, T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<const N: usize, T> FusedIterator for IntoIter<N, T> {}

impl<const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize> IntoIterator
    for SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
{
    type Item = T;
    type IntoIter = IntoIter<N, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.map.into_keys(),
        }
    }
}

impl<'a, const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize> IntoIterator
    for &'a SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, N, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// PartialEq implementation
impl<const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize> PartialEq
    for SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
where
    T: Eq + Hash,
    SH: BuildHasher,
    SI: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().all(|value| other.contains(value))
    }
}

impl<const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize> Eq
    for SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
where
    T: Eq + Hash,
    SH: BuildHasher,
    SI: BuildHasher,
{
}

// Extend implementation
impl<const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize> Extend<T>
    for SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
where
    T: Eq + Hash,
    SH: BuildHasher,
    SI: BuildHasher,
{
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

// FromIterator implementation
impl<const N: usize, T, SH, SI, const LINEAR_THRESHOLD: usize> FromIterator<T>
    for SmallSet<N, T, SH, SI, LINEAR_THRESHOLD>
where
    T: Eq + Hash,
    SH: BuildHasher + Default,
    SI: BuildHasher + Default,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut set: SmallSet<8, i32> = SmallSet::new();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
        assert!(set.is_inline());

        assert_eq!(set.insert(1), true);
        assert_eq!(set.insert(2), true);
        assert_eq!(set.insert(1), false); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(!set.contains(&3));

        assert_eq!(set.remove(&1), true);
        assert_eq!(set.remove(&1), false);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_spill_to_heap() {
        let mut set: SmallSet<4, i32> = SmallSet::new();
        assert!(set.is_inline());

        for i in 0..10 {
            set.insert(i);
        }

        assert!(!set.is_inline());
        assert_eq!(set.len(), 10);

        for i in 0..10 {
            assert!(set.contains(&i));
        }
    }

    #[test]
    fn test_iteration() {
        let mut set: SmallSet<8, i32> = SmallSet::new();
        for i in 0..5 {
            set.insert(i);
        }

        let mut collected: Vec<_> = set.iter().copied().collect();
        collected.sort();
        assert_eq!(collected, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_from_iterator() {
        let set: SmallSet<8, i32> = (0..5).collect();
        assert_eq!(set.len(), 5);
        for i in 0..5 {
            assert!(set.contains(&i));
        }
    }

    #[test]
    fn test_equality() {
        let set1: SmallSet<8, i32> = (0..5).collect();
        let set2: SmallSet<8, i32> = (0..5).collect();
        let set3: SmallSet<8, i32> = (1..6).collect();

        assert_eq!(set1, set2);
        assert_ne!(set1, set3);
    }

    #[test]
    fn test_retain() {
        let mut set: SmallSet<8, i32> = (0..8).collect();
        set.retain(|&x| x % 2 == 0);
        assert_eq!(set.len(), 4);
        assert!(set.contains(&0));
        assert!(set.contains(&2));
        assert!(!set.contains(&1));
        assert!(!set.contains(&3));
    }

    #[test]
    fn test_get() {
        let set: SmallSet<8, _> = [1, 2, 3].into_iter().collect();
        assert_eq!(set.get(&2), Some(&2));
        assert_eq!(set.get(&4), None);
    }

    #[test]
    fn test_take() {
        let mut set: SmallSet<8, i32> = SmallSet::new();
        set.insert(1);
        set.insert(2);
        assert_eq!(set.take(&1), Some(1));
        assert_eq!(set.take(&1), None);
        assert_eq!(set.len(), 1);
        assert!(set.contains(&2));
    }

    #[test]
    fn test_replace() {
        let mut set: SmallSet<8, i32> = SmallSet::new();
        set.insert(1);
        assert_eq!(set.replace(1), Some(1));
        assert_eq!(set.replace(2), None);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_capacity() {
        let set: SmallSet<16, i32> = SmallSet::new();
        assert_eq!(set.capacity(), 16);

        let set: SmallSet<8, i32> = SmallSet::with_capacity(20);
        assert!(set.capacity() >= 20);
        assert!(!set.is_inline());
    }

    #[test]
    fn test_debug() {
        let mut set: SmallSet<8, i32> = SmallSet::new();
        set.insert(1);
        set.insert(2);
        let debug_str = format!("{:?}", set);
        assert!(debug_str.contains('1'));
        assert!(debug_str.contains('2'));
    }

    #[test]
    fn test_extend() {
        let mut set: SmallSet<8, i32> = SmallSet::new();
        set.extend([1, 2, 3]);
        assert_eq!(set.len(), 3);
        set.extend([3, 4, 5]);
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn test_into_iterator() {
        let set: SmallSet<8, i32> = [1, 2, 3].into_iter().collect();
        let mut vec: Vec<_> = set.into_iter().collect();
        vec.sort();
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_clone() {
        let set1: SmallSet<8, i32> = [1, 2, 3].into_iter().collect();
        let set2 = set1.clone();
        assert_eq!(set1, set2);
    }

    #[test]
    fn test_with_hasher() {
        use std::{collections::hash_map::DefaultHasher, hash::BuildHasherDefault};

        let s = BuildHasherDefault::<DefaultHasher>::default();
        let mut set = SmallSet::<8, i32, _>::with_hasher(s);
        set.insert(1);
        assert!(set.contains(&1));
    }

    #[test]
    fn test_clear() {
        let mut set: SmallSet<8, i32> = [1, 2, 3].into_iter().collect();
        assert_eq!(set.len(), 3);
        set.clear();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_insert_duplicates() {
        let mut set: SmallSet<8, i32> = SmallSet::new();
        assert!(set.insert(1));
        assert!(!set.insert(1));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_contains_different_types() {
        let mut set: SmallSet<8, String> = SmallSet::new();
        set.insert("hello".to_string());

        // Test with &str
        assert!(set.contains("hello"));
        assert!(!set.contains("world"));
    }
}
