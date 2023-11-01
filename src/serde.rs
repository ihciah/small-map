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

impl<const N: usize, K, V, H> Serialize for SmallMap<N, K, V, H>
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

impl<'de, const N: usize, K, V, S> Deserialize<'de> for SmallMap<N, K, V, S>
where
    K: Deserialize<'de> + Eq + Hash,
    V: Deserialize<'de>,
    S: BuildHasher + Default,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MapVisitor<const N: usize, K, V, S> {
            marker: PhantomData<SmallMap<N, K, V, S>>,
        }

        impl<'de, const N: usize, K, V, S> Visitor<'de> for MapVisitor<N, K, V, S>
        where
            K: Deserialize<'de> + Eq + Hash,
            V: Deserialize<'de>,
            S: BuildHasher + Default,
        {
            type Value = SmallMap<N, K, V, S>;

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
