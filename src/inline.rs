use core::{
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    mem::{self, transmute, MaybeUninit},
    ptr::NonNull,
};

use crate::{
    raw::{
        h2,
        util::{equivalent_key, likely, make_hash, Bucket, InsertSlot, SizedTypeProperties},
        Group,
    },
    Equivalent,
};

/// Lazy hash computation - computes hash only when needed and caches the result.
enum HashValue<F> {
    Pending(F),
    Computed(u64),
}

impl<F: FnOnce() -> u64> HashValue<F> {
    #[inline]
    fn new(f: F) -> Self {
        Self::Pending(f)
    }

    #[inline]
    fn get(&mut self) -> u64 {
        match self {
            Self::Pending(_) => {
                let f = match core::mem::replace(self, Self::Computed(0)) {
                    Self::Pending(f) => f,
                    Self::Computed(_) => unsafe { core::hint::unreachable_unchecked() },
                };
                let hash = f();
                *self = Self::Computed(hash);
                hash
            }
            Self::Computed(hash) => *hash,
        }
    }
}

#[derive(Clone)]
pub struct Inline<const N: usize, K, V, S, SI> {
    raw: RawInline<N, (K, V)>,
    // Option is for take, S always exists before drop.
    inline_hasher: Option<SI>,
    heap_hasher: Option<S>,
}

struct RawInline<const N: usize, T> {
    aligned_groups: AlignedGroups<N>,
    len: usize,
    data: [MaybeUninit<T>; N],
}

impl<const N: usize, T: Clone> Clone for RawInline<N, T> {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: MaybeUninit doesn't require initialization
        let mut aligned_groups: AlignedGroups<N> = unsafe { MaybeUninit::uninit().assume_init() };
        let mut data: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

        // Only 0..len is valid
        for (i, group) in self.aligned_groups.groups.iter().take(self.len).enumerate() {
            aligned_groups.groups[i] = *group;
            data[i] = MaybeUninit::new(unsafe { self.data[i].assume_init_ref().clone() });
        }

        Self {
            aligned_groups,
            len: self.len,
            data,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct AlignedGroups<const N: usize> {
    groups: [MaybeUninit<u8>; N],
    _align: [Group; 0],
}

impl<const N: usize> AlignedGroups<N> {
    #[inline]
    unsafe fn ctrl(&self, index: usize) -> *mut u8 {
        (self.groups.as_ptr() as *const u8).add(index).cast_mut()
    }
}

impl<const N: usize, T> Drop for RawInline<N, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe { self.drop_elements() }
    }
}

impl<const N: usize, T> RawInline<N, T> {
    #[inline]
    unsafe fn drop_elements(&mut self) {
        if T::NEEDS_DROP {
            // Data is contiguous in 0..len
            for i in 0..self.len {
                core::ptr::drop_in_place(self.data[i].as_mut_ptr());
            }
        }
    }

    /// Threshold for using linear search instead of SIMD.
    const USE_LINEAR_THRESHOLD: usize = Group::WIDTH;

    /// Searches for an element. Uses linear search for small N/len, SIMD otherwise.
    /// Hash is computed lazily only when SIMD path is taken.
    #[inline]
    fn find<F: FnOnce() -> u64>(
        &self,
        hash: &mut HashValue<F>,
        mut eq: impl FnMut(&T) -> bool,
    ) -> Option<Bucket<T>> {
        if N <= Self::USE_LINEAR_THRESHOLD || self.len <= Self::USE_LINEAR_THRESHOLD {
            // Linear search - no hash needed
            for i in 0..self.len {
                if eq(unsafe { self.data.get_unchecked(i).assume_init_ref() }) {
                    return Some(unsafe { self.bucket(i) });
                }
            }
            None
        } else {
            // SIMD search based on len
            unsafe {
                let h2_hash = h2(hash.get());
                let full_groups = self.len / Group::WIDTH;
                let mut probe_pos = 0;

                for _ in 0..full_groups {
                    let group = Group::load(self.aligned_groups.ctrl(probe_pos));
                    for bit in group.match_byte(h2_hash) {
                        let index = probe_pos + bit;
                        if likely(eq(self.data.get_unchecked(index).assume_init_ref())) {
                            return Some(self.bucket(index));
                        }
                    }
                    probe_pos += Group::WIDTH;
                }

                let tail_len = self.len % Group::WIDTH;
                if tail_len > 0 {
                    let group = Group::load(self.aligned_groups.ctrl(probe_pos));
                    for bit in group.match_byte(h2_hash).and(Group::LOWEST_MASK[tail_len]) {
                        let index = probe_pos + bit;
                        if likely(eq(self.data.get_unchecked(index).assume_init_ref())) {
                            return Some(self.bucket(index));
                        }
                    }
                }
                None
            }
        }
    }

    /// Inserts a new element into the table in the given slot, and returns its
    /// raw bucket.
    #[inline]
    unsafe fn insert_in_slot(&mut self, hash: u64, slot: InsertSlot, value: T) -> Bucket<T> {
        self.record_item_insert_at(slot.index, hash);
        let bucket = self.bucket(slot.index);
        bucket.write(value);
        bucket
    }

    /// Records an item insertion at the given index.
    #[inline]
    unsafe fn record_item_insert_at(&mut self, index: usize, hash: u64) {
        self.set_ctrl_h2(index, hash);
        self.len += 1;
    }

    /// Sets a control byte to the hash, and possibly also the replicated control byte at
    /// the end of the array.
    #[inline]
    unsafe fn set_ctrl_h2(&mut self, index: usize, hash: u64) {
        // SAFETY: The caller must uphold the safety rules for the [`RawTableInner::set_ctrl_h2`]
        *self.aligned_groups.ctrl(index) = h2(hash);
    }

    /// Finds and removes an element from the table.
    #[inline]
    fn remove_entry<F: FnOnce() -> u64>(
        &mut self,
        hash: &mut HashValue<F>,
        eq: impl FnMut(&T) -> bool,
    ) -> Option<T> {
        match self.find(hash, eq) {
            Some(bucket) => Some(unsafe { self.remove(bucket).0 }),
            None => None,
        }
    }

    /// Removes an element from the table, returning it.
    /// Uses swap-delete: swaps with last element to maintain contiguity.
    #[inline]
    #[allow(clippy::needless_pass_by_value)]
    unsafe fn remove(&mut self, item: Bucket<T>) -> (T, InsertSlot) {
        let index = self.bucket_index(&item);
        // Read value before swap-delete
        let value = item.read();
        self.erase(index);
        (value, InsertSlot { index })
    }

    /// Returns the index of a bucket from a `Bucket`.
    #[inline]
    unsafe fn bucket_index(&self, bucket: &Bucket<T>) -> usize {
        bucket.to_base_index(NonNull::new_unchecked(self.data.as_ptr() as _))
    }

    /// Erases the element at index using swap-delete.
    /// Copies the last element to index position to maintain contiguous storage.
    /// Caller must ensure the value at index has been moved out or dropped.
    #[inline]
    unsafe fn erase(&mut self, index: usize) {
        let last = self.len - 1;
        if index != last {
            core::ptr::copy_nonoverlapping(
                self.data[last].as_ptr(),
                self.data[index].as_mut_ptr(),
                1,
            );
            self.aligned_groups.groups[index] = self.aligned_groups.groups[last];
        }
        self.len -= 1;
    }

    /// Returns a pointer to an element in the table.
    #[inline]
    unsafe fn bucket(&self, index: usize) -> Bucket<T> {
        Bucket::from_base_index(
            NonNull::new_unchecked(transmute::<*mut MaybeUninit<T>, *mut T>(
                self.data.as_ptr().cast_mut(),
            )),
            index,
        )
    }
}

impl<const N: usize, K, V> RawInline<N, (K, V)> {
    /// Retains only elements where f returns true.
    /// Uses swap-delete to maintain contiguous storage.
    #[inline]
    fn retain<F>(&mut self, f: &mut F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let mut i = 0;
        while i < self.len {
            let (k, v) = unsafe { self.data[i].assume_init_mut() };
            if f(k, v) {
                i += 1;
            } else {
                unsafe {
                    core::ptr::drop_in_place(self.data[i].as_mut_ptr());
                    self.erase(i);
                }
                // Don't increment i, check the swapped element
            }
        }
    }
}

/// Iterator over references to key-value pairs.
pub struct Iter<'a, const N: usize, K, V> {
    data: &'a [MaybeUninit<(K, V)>; N],
    index: usize,
    len: usize,
}

impl<'a, const N: usize, K, V> Iterator for Iter<'a, N, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let kv = unsafe { self.data[self.index].assume_init_ref() };
            self.index += 1;
            Some((&kv.0, &kv.1))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, const N: usize, K, V> ExactSizeIterator for Iter<'a, N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.len - self.index
    }
}

impl<'a, const N: usize, K, V> FusedIterator for Iter<'a, N, K, V> {}

impl<'a, const N: usize, K, V> Clone for Iter<'a, N, K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            index: self.index,
            len: self.len,
        }
    }
}

/// Owning iterator over key-value pairs.
pub struct IntoIter<const N: usize, K, V> {
    data: [MaybeUninit<(K, V)>; N],
    index: usize,
    len: usize,
}

impl<const N: usize, K, V> Iterator for IntoIter<N, K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let kv = unsafe { self.data[self.index].as_ptr().read() };
            self.index += 1;
            Some(kv)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<const N: usize, K, V> ExactSizeIterator for IntoIter<N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.len - self.index
    }
}

impl<const N: usize, K, V> FusedIterator for IntoIter<N, K, V> {}

impl<const N: usize, K, V> Drop for IntoIter<N, K, V> {
    fn drop(&mut self) {
        // Drop remaining elements
        for i in self.index..self.len {
            unsafe { core::ptr::drop_in_place(self.data[i].as_mut_ptr()) };
        }
    }
}

impl<const N: usize, K, V, S, SI> IntoIterator for Inline<N, K, V, S, SI> {
    type Item = (K, V);
    type IntoIter = IntoIter<N, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let iter = IntoIter {
            data: unsafe { core::ptr::read(&self.raw.data) },
            index: 0,
            len: self.raw.len,
        };
        mem::forget(self);
        iter
    }
}

impl<const N: usize, K, V, S, SI> Inline<N, K, V, S, SI> {
    #[inline]
    pub(crate) fn iter(&self) -> Iter<'_, N, K, V> {
        Iter {
            data: &self.raw.data,
            index: 0,
            len: self.raw.len,
        }
    }

    #[inline]
    pub(crate) const fn new(hash_builder: SI, heap_hasher: S) -> Self {
        assert!(N != 0, "SmallMap cannot be initialized with zero size.");
        Self {
            raw: RawInline {
                // SAFETY: MaybeUninit doesn't require initialization, len tracks validity
                aligned_groups: unsafe { MaybeUninit::uninit().assume_init() },
                len: 0,
                data: unsafe { MaybeUninit::uninit().assume_init() },
            },
            inline_hasher: Some(hash_builder),
            heap_hasher: Some(heap_hasher),
        }
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.raw.len == 0
    }

    #[inline]
    pub(crate) fn is_full(&self) -> bool {
        self.raw.len == N
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.raw.len
    }

    // # Safety
    // Hasher must exist.
    #[inline]
    pub(crate) unsafe fn take_inline_hasher(&mut self) -> SI {
        self.inline_hasher.take().unwrap_unchecked()
    }

    // # Safety
    // Hasher must exist.
    #[inline]
    pub(crate) unsafe fn take_heap_hasher(&mut self) -> S {
        self.heap_hasher.take().unwrap_unchecked()
    }

    #[inline]
    fn inline_hasher(&self) -> &SI {
        self.inline_hasher.as_ref().unwrap()
    }
}

impl<const N: usize, K, V, S, SI> Inline<N, K, V, S, SI>
where
    K: Eq + Hash,
    SI: BuildHasher,
{
    /// Returns a reference to the value corresponding to the key.
    #[inline]
    pub(crate) fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner(k) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    /// Returns a reference to the value corresponding to the key.
    #[inline]
    pub(crate) fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner_mut(k) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    /// Returns the key-value pair corresponding to the supplied key.
    #[inline]
    pub(crate) fn get_key_value<Q>(&self, k: &Q) -> Option<(&K, &V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner(k) {
            Some((key, value)) => Some((key, value)),
            None => None,
        }
    }

    /// Inserts a key-value pair into the map.
    #[inline]
    pub(crate) fn insert(&mut self, k: K, v: V) -> Option<V> {
        let hash_builder = self.inline_hasher.as_ref().unwrap();
        let mut hash = HashValue::new(|| make_hash::<K, SI>(hash_builder, &k));
        match self.raw.find(&mut hash, equivalent_key(&k)) {
            Some(bucket) => Some(mem::replace(unsafe { &mut bucket.as_mut().1 }, v)),
            None => {
                let slot = InsertSlot {
                    index: self.raw.len,
                };
                unsafe { self.raw.insert_in_slot(hash.get(), slot, (k, v)) };
                None
            }
        }
    }

    /// Inserts a key-value pair into the map without checking if the key already exists.
    ///
    /// # Safety
    /// The caller must ensure that the key does not already exist in the map.
    #[inline]
    pub(crate) unsafe fn insert_unique_unchecked(&mut self, k: K, v: V) -> (&K, &mut V) {
        let hash_builder = self.inline_hasher.as_ref().unwrap();
        let hash = make_hash::<K, SI>(hash_builder, &k);
        let slot = InsertSlot {
            index: self.raw.len,
        };
        let bucket = self.raw.insert_in_slot(hash, slot, (k, v));
        let (k, v) = bucket.as_mut();
        (k, v)
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map. Keeps the allocated memory for reuse.
    #[inline]
    pub(crate) fn remove_entry<Q>(&mut self, k: &Q) -> Option<(K, V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        let hash_builder = self.inline_hasher.as_ref().unwrap();
        let mut hash = HashValue::new(|| make_hash::<Q, SI>(hash_builder, k));
        self.raw.remove_entry(&mut hash, equivalent_key(k))
    }

    /// Retains only the elements specified by the predicate.
    #[inline]
    pub(crate) fn retain<F>(&mut self, f: &mut F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.raw.retain(f);
    }

    #[inline]
    fn get_inner<Q>(&self, k: &Q) -> Option<&(K, V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        if self.is_empty() {
            return None;
        }
        let mut hash = HashValue::new(|| make_hash::<Q, SI>(self.inline_hasher(), k));
        match self.raw.find(&mut hash, equivalent_key(k)) {
            Some(bucket) => Some(unsafe { bucket.as_ref() }),
            None => None,
        }
    }

    #[inline]
    fn get_inner_mut<Q>(&mut self, k: &Q) -> Option<&mut (K, V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        if self.is_empty() {
            return None;
        }
        let hash_builder = self.inline_hasher.as_ref().unwrap();
        let mut hash = HashValue::new(|| make_hash::<Q, SI>(hash_builder, k));
        match self.raw.find(&mut hash, equivalent_key(k)) {
            Some(bucket) => Some(unsafe { bucket.as_mut() }),
            None => None,
        }
    }
}
