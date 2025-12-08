# Guide to Benchmark

Command:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench simple
```

Result can be found under `target/criterion/simple`.

To generate the chart link shown in README, run:

```bash
cd benches
./gen_charts.sh
```
