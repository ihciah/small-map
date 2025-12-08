# Small-Map

[![ci](https://github.com/ihciah/small-map/actions/workflows/ci.yml/badge.svg)](https://github.com/ihciah/small-map/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/small-map.svg)](https://crates.io/crates/small-map)
[![docs.rs](https://img.shields.io/docsrs/small-map)](https://docs.rs/small-map/latest/small_map/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/ihciah/small-map/blob/master/LICENSE-MIT)
[![License](https://img.shields.io/badge/license-Apache-green.svg)](https://github.com/ihciah/small-map/blob/master/LICENSE-APACHE)


An inline SIMD accelerated hashmap designed for small amount of data.

## Usage

```rust
use small_map::FxSmallMap;
// Don't worry about the 32 here, it's the inline size, not the limitation.
// If the size grows more than 32, it will automatically switch to heap impl.
type MySmallMap<K, V> = FxSmallMap<32, K, V>;

let mut map = MySmallMap::new();
map.insert(1_u8, 2_u8);
assert_eq!(*map.get(&1).unwrap(), 2);
```

Usually you can use this map for short lifetime and small key/values, for example, http or RPC headers.

You can use it like other normal hashmaps without worrying about its size.

## Choosing N

The choice of `N` involves a trade-off between inline storage benefits and struct size overhead:

**Struct size** = `N * (1 + sizeof(K) + sizeof(V)) + 2 * sizeof(usize) + sizeof(Hasher)`

### Considerations

1. **Key/Value size**: Larger K/V types increase struct size proportionally with N
2. **Pass frequency**: If the map is frequently passed by value (moved/copied), a large struct hurts performance
3. **Expected element count**: N should cover most use cases to avoid heap fallback

**Tip**: Choose N as a multiple of SIMD_WIDTH (16 on x86_64, 8 on aarch64) for optimal SIMD utilization. When uncertain, start with `N = 16` or `N = 32`.

## Performance

The performance of SmallMap depends on the capacity, Key/Value size and operation scenario.

It is recommended to set the size to 16(or 32) because with SSE2 it can search 16 items within a single instruction. It is only recommended to use SmallMap for small Key/Values, such as numbers and strings.

![Map Performance (4-32)](https://quickchart.io/chart?c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B4%2C12%2C20%2C32%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22small-map%20%28FxHash%29%22%2C%22data%22%3A%5B12.32%2C57.99%2C123.09%2C183.09%5D%2C%22borderColor%22%3A%22%234CAF50%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22FxHashMap%22%2C%22data%22%3A%5B63.33%2C154.8%2C171.67%2C191.99%5D%2C%22borderColor%22%3A%22%232196F3%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22micromap%22%2C%22data%22%3A%5B8.81%2C55.2%2C138.82%2C367.95%5D%2C%22borderColor%22%3A%22%23FF9800%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%5D%7D%2C%22options%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Map%20Performance%20%284-32%29%22%7D%2C%22scales%22%3A%7B%22yAxes%22%3A%5B%7B%22scaleLabel%22%3A%7B%22display%22%3Atrue%2C%22labelString%22%3A%22Time%20%28ns%29%22%7D%7D%5D%2C%22xAxes%22%3A%5B%7B%22scaleLabel%22%3A%7B%22display%22%3Atrue%2C%22labelString%22%3A%22Size%22%7D%7D%5D%7D%7D%7D)

![Map Performance (4-64)](https://quickchart.io/chart?c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B4%2C12%2C20%2C32%2C48%2C64%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22small-map%20%28FxHash%29%22%2C%22data%22%3A%5B12.32%2C57.99%2C123.09%2C183.09%2C281.29%2C337.03%5D%2C%22borderColor%22%3A%22%234CAF50%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22FxHashMap%22%2C%22data%22%3A%5B63.33%2C154.8%2C171.67%2C191.99%2C290.65%2C342.39%5D%2C%22borderColor%22%3A%22%232196F3%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22micromap%22%2C%22data%22%3A%5B8.81%2C55.2%2C138.82%2C367.95%2C882.87%2C1511.4%5D%2C%22borderColor%22%3A%22%23FF9800%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%5D%7D%2C%22options%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Map%20Performance%20%284-64%29%22%7D%2C%22scales%22%3A%7B%22yAxes%22%3A%5B%7B%22scaleLabel%22%3A%7B%22display%22%3Atrue%2C%22labelString%22%3A%22Time%20%28ns%29%22%7D%7D%5D%2C%22xAxes%22%3A%5B%7B%22scaleLabel%22%3A%7B%22display%22%3Atrue%2C%22labelString%22%3A%22Size%22%7D%7D%5D%7D%7D%7D)

![All Maps Performance (4-128)](https://quickchart.io/chart?c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B4%2C8%2C16%2C32%2C64%2C96%2C128%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22small-map%20%28FxHash%29%22%2C%22data%22%3A%5B12.1%2C30.43%2C93.51%2C178.11%2C340.9%2C509.67%2C652.58%5D%2C%22borderColor%22%3A%22%234CAF50%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22FxHashMap%22%2C%22data%22%3A%5B60.28%2C107.08%2C142.33%2C188.8%2C339.17%2C504.71%2C632.27%5D%2C%22borderColor%22%3A%22%232196F3%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22micromap%22%2C%22data%22%3A%5B8.73%2C24.99%2C95.65%2C354.12%2C1545.18%2C3439.17%2C5269.52%5D%2C%22borderColor%22%3A%22%23FF9800%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22hashbrown%22%2C%22data%22%3A%5B72.07%2C137.2%2C253.01%2C485.81%2C959.14%2C1487.69%2C1934.55%5D%2C%22borderColor%22%3A%22%239C27B0%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22std%20HashMap%22%2C%22data%22%3A%5B89.84%2C176.52%2C328.57%2C644.75%2C1274.57%2C1965.68%2C2551.81%5D%2C%22borderColor%22%3A%22%23607D8B%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22BTreeMap%22%2C%22data%22%3A%5B27.17%2C63.19%2C179.84%2C417.64%2C1032.9%2C1746.43%2C2459.97%5D%2C%22borderColor%22%3A%22%23795548%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22litemap%22%2C%22data%22%3A%5B22.25%2C79.44%2C194.53%2C489.46%2C1161.33%2C1948.89%2C2700.7%5D%2C%22borderColor%22%3A%22%23E91E63%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%2C%7B%22label%22%3A%22tinymap%22%2C%22data%22%3A%5B24.08%2C95.06%2C411.28%2C1651.5%2C7825.83%2Cnull%2Cnull%5D%2C%22borderColor%22%3A%22%2300BCD4%22%2C%22fill%22%3Afalse%2C%22spanGaps%22%3Atrue%7D%5D%7D%2C%22options%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22All%20Maps%20Performance%20%284-128%29%22%7D%2C%22scales%22%3A%7B%22yAxes%22%3A%5B%7B%22scaleLabel%22%3A%7B%22display%22%3Atrue%2C%22labelString%22%3A%22Time%20%28ns%29%22%7D%7D%5D%2C%22xAxes%22%3A%5B%7B%22scaleLabel%22%3A%7B%22display%22%3Atrue%2C%22labelString%22%3A%22Size%22%7D%7D%5D%7D%7D%7D)

> [!NOTE]
> The lower the time(ns), the better the performance. You can find the benchmark source code in the `benches` folder.
>
> Benchmark with u8 key/value on AMD Ryzen 9 7950X.

## How it Works

Like SmallVec, for HashMap with a small amount of data, inlining the data can avoid heap allocation overhead and improve performance.

SmallMap stores key-value pairs in a contiguous inline array, with an additional control byte array storing the hash fingerprint for SIMD-accelerated lookup.

### Lookup Strategy

The lookup strategy is determined at compile time based on `N` and at runtime based on current length:

1. **Linear Search** (no hash computation): When `N < SIMD_WIDTH` or `len < SIMD_WIDTH`
   - Simply iterates through the array comparing keys directly
   - Avoids hash computation overhead for small maps
   - SIMD_WIDTH is 16 on x86_64 (SSE2) and 8 on aarch64 (NEON)

2. **SIMD Search** (with h2 hash): When `N >= SIMD_WIDTH` and `len >= SIMD_WIDTH`
   - Computes h2 (8-bit hash fingerprint) from the key's hash
   - Uses SIMD to compare h2 against 16 or 8 control bytes in parallel
   - Only performs key comparison on h2 matches

3. **Heap Fallback**: When `len > N`
   - Data is moved to a hashbrown-based heap HashMap
   - Performance is equivalent to hashbrown

### Memory Layout

```text
┌─────────────────────────────────────┐
│  AlignedGroups (N bytes)            │  ← h2 control bytes for SIMD
├─────────────────────────────────────┤
│  len: usize                         │
├─────────────────────────────────────┤
│  data: [(K, V); N]                  │  ← contiguous key-value pairs
└─────────────────────────────────────┘
```

### Complexity

Let `W` = SIMD_WIDTH (16 on x86_64, 8 on aarch64)

| Operation | len ≤ W | W < len ≤ N | len > N |
|-----------|---------|-------------|---------|
| `get` | O(len) | O(len/W) | O(1)* |
| `insert` | O(len) | O(len/W) | O(1)* |
| `insert_unique_unchecked` | O(1) | O(1) | O(1)* |
| `remove` | O(len) | O(len/W) | O(1)* |
| `iter` | O(len) | O(len) | O(capacity) |

\* amortized

## Credit

[Hashbrown](https://crates.io/crates/hashbrown) is heavily referenced to and copied by this project, and is a very elegant and efficient implementation.
