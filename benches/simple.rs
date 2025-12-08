use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

macro_rules! get_set {
    ($map:ident, $n:expr) => {
        for i in 0..$n {
            $map.insert(i, i);
        }
        for i in 0..$n {
            black_box($map.get(&i));
        }
    };
}

#[inline(always)]
fn smallmap<const N: usize, H: std::hash::BuildHasher + Default>(n: u8) {
    let mut map = small_map::SmallMap::<N, _, _, H>::with_capacity(n as usize);
    get_set!(map, n);
}

#[inline(always)]
fn hashbrown(n: u8) {
    use std::collections::hash_map::RandomState;
    let mut map = hashbrown::HashMap::<_, _, RandomState>::with_capacity_and_hasher(
        n as usize,
        RandomState::default(),
    );
    get_set!(map, n);
}

#[inline(always)]
fn std_btreemap(n: u8) {
    let mut map = std::collections::BTreeMap::<_, _>::new();
    get_set!(map, n);
}

#[inline(always)]
fn std_hashmap(n: u8) {
    let mut map =
        std::collections::HashMap::<_, _, std::collections::hash_map::RandomState>::with_capacity(
            n as usize,
        );
    get_set!(map, n);
}

#[inline(always)]
fn fx_hashmap(n: u8) {
    let mut map =
        std::collections::HashMap::with_capacity_and_hasher(n as usize, rustc_hash::FxBuildHasher);
    get_set!(map, n);
}

#[inline(always)]
fn micromap<const N: usize>(n: u8) {
    let mut map = micromap::Map::<u8, u8, N>::new();
    get_set!(map, n);
}

#[inline(always)]
fn tinymap<const N: usize>(n: u8) {
    let mut map = tinymap::array_map::ArrayMap::<u8, u8, N>::new();
    get_set!(map, n);
}

#[inline(always)]
fn litemap(n: u8) {
    let mut map = litemap::LiteMap::<u8, u8>::with_capacity(n as usize);
    get_set!(map, n);
}

fn criterion_benchmark(c: &mut Criterion) {
    let micromap_threshold = 128;
    let mut tinymap_threshold = 64;
    macro_rules! bench_size {
        ($group: expr, $size: expr) => {
            bench_size!($group, $size, $size);
        };
        ($group: expr, $size: expr, $N: expr) => {
            let size = $size;
            $group.bench_with_input(
                BenchmarkId::new("small-map::FxSmallMap ⬅", size),
                &size,
                |b, &n| b.iter(|| smallmap::<$N, rustc_hash::FxBuildHasher>(n)),
            );
            // Not run MicroMap for bigger size, as it gets very big
            if size <= micromap_threshold {
                $group.bench_with_input(BenchmarkId::new("micromap::Map", size), &size, |b, &n| {
                    b.iter(|| micromap::<$size>(n))
                });
            }
            // Not run ArrayMap for bigger size, as it gets very big
            if size <= tinymap_threshold {
                $group.bench_with_input(
                    BenchmarkId::new("tinymap::array_map::ArrayMap", size),
                    &size,
                    |b, &n| b.iter(|| tinymap::<$size>(n)),
                );
            }
            $group.bench_with_input(
                BenchmarkId::new("litemap::LiteMap", size),
                &size,
                |b, &n| b.iter(|| litemap(n)),
            );
            $group.bench_with_input(
                BenchmarkId::new("hashbrown::HashMap", size),
                &size,
                |b, &n| b.iter(|| hashbrown(n)),
            );
            $group.bench_with_input(
                BenchmarkId::new("std::collections::HashMap", size),
                &size,
                |b, &n| b.iter(|| std_hashmap(n)),
            );
            $group.bench_with_input(
                BenchmarkId::new("rustc_hash::FxHashMap", size),
                &size,
                |b, &n| b.iter(|| fx_hashmap(n)),
            );
            $group.bench_with_input(
                BenchmarkId::new("std::collections::BTreeMap", size),
                &size,
                |b, &n| b.iter(|| std_btreemap(n)),
            );
        };
    }

    let mut big_group = c.benchmark_group("Maps Benchmark in 0~196 Scale");
    bench_size!(big_group, 4);
    bench_size!(big_group, 8);
    bench_size!(big_group, 16);
    bench_size!(big_group, 32, 20);
    bench_size!(big_group, 64, 20);
    bench_size!(big_group, 96, 20);
    bench_size!(big_group, 128, 20);
    bench_size!(big_group, 192, 20);
    big_group.finish();

    tinymap_threshold = 32;
    let mut small_group = c.benchmark_group("Maps Benchmark in 0~64 Scale");
    bench_size!(small_group, 4);
    bench_size!(small_group, 8);
    bench_size!(small_group, 12);
    bench_size!(small_group, 16);
    bench_size!(small_group, 20);
    bench_size!(small_group, 32, 20);
    bench_size!(small_group, 48, 20);
    bench_size!(small_group, 64, 20);
    small_group.finish();
}

#[cfg(unix)]
mod profile {
    use std::{fs::File, path::Path};

    use criterion::profiler::Profiler;
    use pprof::ProfilerGuard;

    pub struct FlamegraphProfiler<'a> {
        frequency: core::ffi::c_int,
        active_profiler: Option<ProfilerGuard<'a>>,
    }

    impl<'a> FlamegraphProfiler<'a> {
        #[allow(dead_code)]
        pub fn new(frequency: core::ffi::c_int) -> Self {
            FlamegraphProfiler {
                frequency,
                active_profiler: None,
            }
        }
    }

    impl<'a> Profiler for FlamegraphProfiler<'a> {
        fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
            self.active_profiler = Some(ProfilerGuard::new(self.frequency).unwrap());
        }

        fn stop_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
            std::fs::create_dir_all(benchmark_dir).unwrap();
            let flamegraph_path = benchmark_dir.join("flamegraph.svg");
            let flamegraph_file = File::create(flamegraph_path)
                .expect("File system error while creating flamegraph.svg");
            if let Some(profiler) = self.active_profiler.take() {
                profiler
                    .report()
                    .build()
                    .unwrap()
                    .flamegraph(flamegraph_file)
                    .expect("Error writing flamegraph");
            }
        }
    }
}

criterion_main!(benches);
criterion_group!(benches, criterion_benchmark);
// criterion_group! {
//     name = benches;
//     // This can be any expression that returns a `Criterion` object.
//     config = Criterion::default().with_profiler(profile::FlamegraphProfiler::new(100));
//     targets = criterion_benchmark
// }
