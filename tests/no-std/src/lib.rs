#![no_std]

use core::hash::{BuildHasher, Hasher};

use small_map::{SmallMap, SmallSet};

pub struct TestHasher(u64);

impl Hasher for TestHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

#[derive(Clone, Default)]
pub struct TestBuildHasher;

impl BuildHasher for TestBuildHasher {
    type Hasher = TestHasher;

    fn build_hasher(&self) -> Self::Hasher {
        TestHasher(0xcbf29ce484222325)
    }
}

pub type TestMap = SmallMap<2, u32, u32, TestBuildHasher, TestBuildHasher>;
pub type TestSet = SmallSet<2, u32, TestBuildHasher, TestBuildHasher>;

pub fn exercise_map() -> TestMap {
    let mut map = TestMap::with_hashers(TestBuildHasher, TestBuildHasher);
    for i in 0..4 {
        map.insert(i, i * 2);
    }
    let _ = map.get(&1);
    map.remove(&0);
    map
}

pub fn exercise_set() -> TestSet {
    let mut set = TestSet::with_hashers(TestBuildHasher, TestBuildHasher);
    for i in 0..4 {
        set.insert(i);
    }
    let _ = set.contains(&1);
    set.remove(&0);
    set
}

#[cfg(feature = "fxhash")]
pub fn exercise_fxhash() -> small_map::FxSmallMap<2, u32, u32> {
    let mut map = small_map::FxSmallMap::new();
    for i in 0..4 {
        map.insert(i, i * 2);
    }
    map
}

#[cfg(feature = "rapidhash")]
pub fn exercise_rapidhash() -> small_map::RapidSmallMap<2, u32, u32> {
    let mut map = small_map::RapidSmallMap::new();
    for i in 0..4 {
        map.insert(i, i * 2);
    }
    map
}

#[cfg(feature = "ahash")]
pub fn exercise_ahash() -> small_map::ASmallMap<2, u32, u32> {
    let mut map = small_map::ASmallMap::new();
    for i in 0..4 {
        map.insert(i, i * 2);
    }
    map
}

#[cfg(feature = "serde")]
pub fn assert_serde_impls() {
    fn assert_impl<T>()
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
    }

    assert_impl::<TestMap>();
}
