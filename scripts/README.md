# Scripts

## `plot_bench.py`

This script allow to plot benchmarks results.

The following dependencies are needed:

* pandas
* bokeh

To run it:

```bash
scripts/plot_bench.py bench_result_1.json ... bench_result_N.json
```

Where `bench_result_i.json` is a benchmark result.

To generate a benchmark result for the cholesky block sparse benchmark:

```bash
build/tests/bench-cpp-cholesky-block-sparse-bench --benchmark_out_format=json --benchmark_out=benchmark_result_i.json
```
