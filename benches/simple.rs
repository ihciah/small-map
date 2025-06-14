use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

fn smallmap<const N: usize>(n: u8) {
    let mut map = small_map::SmallMap::<N, _, _>::default();
    for i in 0..n {
        map.insert(i, i);
    }
    for i in 0..n {
        black_box(map.get(&i));
    }
}

fn hashbrown(n: u8) {
    use std::collections::hash_map::RandomState;
    let mut map = hashbrown::HashMap::<_, _, RandomState>::with_capacity_and_hasher(
        n as usize,
        RandomState::default(),
    );
    for i in 0..n {
        map.insert(i, i);
    }
    for i in 0..n {
        black_box(map.get(&i));
    }
}

fn std_hashmap(n: u8) {
    let mut map =
        std::collections::HashMap::<_, _, std::collections::hash_map::RandomState>::with_capacity(
            n as usize,
        );
    for i in 0..n {
        map.insert(i, i);
    }
    for i in 0..n {
        black_box(map.get(&i));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("smallmap-simple-4", |b| b.iter(|| smallmap::<16>(4)));
    c.bench_function("hashbrown-simple-4", |b| b.iter(|| hashbrown(4)));
    c.bench_function("stdhashmap-simple-4", |b| b.iter(|| std_hashmap(4)));

    c.bench_function("smallmap-simple-8", |b| b.iter(|| smallmap::<16>(8)));
    c.bench_function("hashbrown-simple-8", |b| b.iter(|| hashbrown(8)));
    c.bench_function("stdhashmap-simple-8", |b| b.iter(|| std_hashmap(8)));

    c.bench_function("smallmap-simple-12", |b| b.iter(|| smallmap::<16>(12)));
    c.bench_function("hashbrown-simple-12", |b| b.iter(|| hashbrown(12)));
    c.bench_function("stdhashmap-simple-12", |b| b.iter(|| std_hashmap(12)));

    c.bench_function("smallmap-simple-16", |b| b.iter(|| smallmap::<16>(16)));
    c.bench_function("hashbrown-simple-16", |b| b.iter(|| hashbrown(16)));
    c.bench_function("stdhashmap-simple-16", |b| b.iter(|| std_hashmap(16)));
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
//#[cfg(unix)]
// criterion_group! {
//     name = benches;
//     // This can be any expression that returns a `Criterion` object.
//     config = Criterion::default().with_profiler(profile::FlamegraphProfiler::new(100));
//     targets = criterion_benchmark
// }
