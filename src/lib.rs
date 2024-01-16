#![allow(clippy::manual_map)]

use core::{
    hash::{BuildHasher, Hash},
    hint::unreachable_unchecked,
    iter::FusedIterator,
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

#[derive(Clone)]
pub enum SmallMap<const N: usize, K, V, S = RandomState> {
    Heap(HashMap<K, V, S>),
    Inline(Inline<N, K, V, S>),
}

impl<const N: usize, K, V, S> Debug for SmallMap<N, K, V, S>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<const N: usize, K, V, S: Default> Default for SmallMap<N, K, V, S> {
    /// Creates an empty `SmallMap<N, K, V, S>`, with the `Default` value for the hasher.
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
        Self::with_hasher(Default::default())
    }
}

impl<const N: usize, K, V, S: Default> SmallMap<N, K, V, S> {
    /// Creates an empty `SmallMap`.
    #[inline]
    pub fn new() -> Self {
        Self::with_hasher(Default::default())
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
            Self::Inline(Inline::new(Default::default()))
        }
    }
}

impl<const N: usize, K, V, S> SmallMap<N, K, V, S> {
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
    pub const fn with_hasher(hash_builder: S) -> Self {
        Self::Inline(Inline::new(hash_builder))
    }

    /// Creates an empty `SmallMap` with the specified capacity, using `hash_builder`
    /// to hash the keys.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is smaller than or eq to N, the hash map will not allocate.
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        if capacity > N {
            Self::Heap(HashMap::with_capacity_and_hasher(capacity, hash_builder))
        } else {
            Self::Inline(Inline::new(hash_builder))
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

impl<const N: usize, K, V, S> SmallMap<N, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    #[inline]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        Q: Hash + Equivalent<K>,
    {
        match self {
            SmallMap::Heap(inner) => inner.get(k),
            SmallMap::Inline(inner) => inner.get(k),
        }
    }

    /// Returns a mutable reference to the value corresponding to the key.
    #[inline]
    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        Q: Hash + Equivalent<K>,
    {
        match self {
            SmallMap::Heap(inner) => inner.get_mut(k),
            SmallMap::Inline(inner) => inner.get_mut(k),
        }
    }

    /// Returns the key-value pair corresponding to the supplied key.
    #[inline]
    pub fn get_key_value<Q: ?Sized>(&self, k: &Q) -> Option<(&K, &V)>
    where
        Q: Hash + Equivalent<K>,
    {
        match self {
            SmallMap::Heap(inner) => inner.get_key_value(k),
            SmallMap::Inline(inner) => inner.get_key_value(k),
        }
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self {
            SmallMap::Heap(inner) => inner.insert(key, value),
            SmallMap::Inline(inner) => {
                if raw::util::likely(!inner.is_full()) {
                    return inner.insert(key, value);
                }
                let heap = HashMap::with_capacity_and_hasher(N + 1, unsafe { inner.take_hasher() });

                let heap = match core::mem::replace(self, SmallMap::Heap(heap)) {
                    SmallMap::Heap(_) => unsafe { unreachable_unchecked() },
                    SmallMap::Inline(inline) => {
                        let heap = match self {
                            SmallMap::Heap(heap) => heap,
                            SmallMap::Inline(_) => unsafe { unreachable_unchecked() },
                        };
                        for (k, v) in inline.into_iter() {
                            heap.insert_unique_unchecked(k, v);
                        }
                        heap
                    }
                };
                heap.insert(key, value)
            }
        }
    }

    #[inline]
    pub fn remove<Q: ?Sized>(&mut self, k: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K>,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.remove_entry(k) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    #[inline]
    pub fn remove_entry<Q: ?Sized>(&mut self, k: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K>,
    {
        match self {
            SmallMap::Heap(inner) => inner.remove_entry(k),
            SmallMap::Inline(inner) => inner.remove_entry(k),
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

impl<const N: usize, K, V, S> IntoIterator for SmallMap<N, K, V, S> {
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

impl<'a, const N: usize, K, V, S> IntoIterator for &'a SmallMap<N, K, V, S> {
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

impl<const N: usize, K, V, S> SmallMap<N, K, V, S> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            SmallMap::Heap(inner) => inner.len(),
            SmallMap::Inline(inner) => inner.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            SmallMap::Heap(inner) => inner.is_empty(),
            SmallMap::Inline(inner) => inner.is_empty(),
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, N, K, V> {
        match self {
            SmallMap::Heap(inner) => Iter::Heap(inner.iter()),
            SmallMap::Inline(inner) => Iter::Inline(inner.iter()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use ahash::RandomState;
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
                let mut rng = rand::thread_rng();

                let choice: u8 = rng.gen();
                match choice % 4 {
                    0 => Operation::Insert(rng.gen_range(0..32), rng.gen()),
                    1 => Operation::Remove(rng.gen_range(0..32)),
                    2 => Operation::Get(rng.gen_range(0..32)),
                    3 => Operation::ModifyIfExist(rng.gen_range(0..32), rng.gen()),
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
}
