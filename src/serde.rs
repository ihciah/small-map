use core::{
    fmt,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
};

use serde::{
    de::{MapAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::SmallMap;

mod size_hint {
    use core::cmp;

    /// This presumably exists to prevent denial of service attacks.
    ///
    /// Original discussion: https://github.com/serde-rs/serde/issues/1114.
    #[inline]
    pub(super) fn cautious(hint: Option<usize>) -> usize {
        cmp::min(hint.unwrap_or(0), 4096)
    }
}

impl<const N: usize, K, V, H, SI, const LINEAR_THRESHOLD: usize> Serialize
    for SmallMap<N, K, V, H, SI, LINEAR_THRESHOLD>
where
    K: Serialize + Eq + Hash,
    V: Serialize,
    H: BuildHasher,
{
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_map(self)
    }
}

impl<'de, const N: usize, K, V, S, SI, const LINEAR_THRESHOLD: usize> Deserialize<'de>
    for SmallMap<N, K, V, S, SI, LINEAR_THRESHOLD>
where
    K: Deserialize<'de> + Eq + Hash,
    V: Deserialize<'de>,
    S: BuildHasher + Default,
    SI: BuildHasher + Default,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MapVisitor<const N: usize, K, V, S, SI, const LINEAR_THRESHOLD: usize> {
            marker: PhantomData<SmallMap<N, K, V, S, SI, LINEAR_THRESHOLD>>,
        }

        impl<'de, const N: usize, K, V, S, SI, const LINEAR_THRESHOLD: usize> Visitor<'de>
            for MapVisitor<N, K, V, S, SI, LINEAR_THRESHOLD>
        where
            K: Deserialize<'de> + Eq + Hash,
            V: Deserialize<'de>,
            S: BuildHasher + Default,
            SI: BuildHasher + Default,
        {
            type Value = SmallMap<N, K, V, S, SI, LINEAR_THRESHOLD>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a map")
            }

            #[cfg_attr(feature = "inline-more", inline)]
            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut values = SmallMap::with_capacity_and_hasher(
                    size_hint::cautious(map.size_hint()),
                    S::default(),
                );

                while let Some((key, value)) = map.next_entry()? {
                    values.insert(key, value);
                }

                Ok(values)
            }
        }

        let visitor = MapVisitor {
            marker: PhantomData,
        };
        deserializer.deserialize_map(visitor)
    }
}

#[cfg(test)]
mod tests {
    use core::hash::BuildHasherDefault;
    use std::{collections::hash_map::RandomState, hash::DefaultHasher};

    use crate::SmallMap;

    type TestHasher = BuildHasherDefault<DefaultHasher>;
    type TestMap = SmallMap<4, u32, u32, RandomState, TestHasher, 1>;

    fn roundtrip<T>(map: &T) -> T
    where
        T: ::serde::Serialize + ::serde::de::DeserializeOwned,
    {
        let json = serde_json::to_string(map).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn custom_inline_hasher_inline_roundtrip() {
        let mut map = TestMap::new();
        map.insert(1, 2);
        map.insert(3, 4);
        assert!(map.is_inline());

        let deserialized = roundtrip(&map);
        assert!(deserialized.is_inline());
        assert_eq!(deserialized, map);
    }

    #[test]
    fn custom_inline_hasher_heap_roundtrip() {
        let mut map = TestMap::new();
        for i in 0..8 {
            map.insert(i, i * 2);
        }
        assert!(!map.is_inline());

        let deserialized = roundtrip(&map);
        assert!(!deserialized.is_inline());
        assert_eq!(deserialized, map);
    }

    #[test]
    fn direct_heap_deserialization_does_not_construct_inline_hasher() {
        use core::hash::BuildHasher;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static INLINE_DEFAULTS: AtomicUsize = AtomicUsize::new(0);

        struct CountingInlineHasher;
        impl Default for CountingInlineHasher {
            fn default() -> Self {
                INLINE_DEFAULTS.fetch_add(1, Ordering::SeqCst);
                Self
            }
        }
        impl BuildHasher for CountingInlineHasher {
            type Hasher = DefaultHasher;

            fn build_hasher(&self) -> Self::Hasher {
                DefaultHasher::new()
            }
        }

        let value = serde_json::to_value(
            (0..8)
                .map(|i| (i, i * 2))
                .collect::<std::collections::BTreeMap<_, _>>(),
        )
        .unwrap();
        // `Value` retains its map length, so this uses the direct-heap path.
        let deserialized: SmallMap<4, u32, u32, RandomState, CountingInlineHasher> =
            serde_json::from_value(value).unwrap();

        assert_eq!(deserialized.len(), 8);
        assert!(!deserialized.is_inline());
        assert_eq!(INLINE_DEFAULTS.load(Ordering::SeqCst), 0);
    }

    #[cfg(not(feature = "std"))]
    #[test]
    fn default_hashers_inline_and_heap_roundtrip() {
        for len in [2u32, 8] {
            let mut map = SmallMap::<4, u32, u32>::new();
            for i in 0..len {
                map.insert(i, i * 2);
            }

            let deserialized = roundtrip(&map);
            assert_eq!(map.is_inline(), len <= 4);
            assert_eq!(deserialized.is_inline(), map.is_inline());
            assert_eq!(deserialized, map);
        }
    }
}
