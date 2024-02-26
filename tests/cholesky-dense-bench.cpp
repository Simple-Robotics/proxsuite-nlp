/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#include <cstdint>

#include "proxsuite-nlp/linalg/bunchkaufman.hpp"
#include "proxsuite-nlp/linalg/dense.hpp"
#include "util.hpp"

#include <benchmark/benchmark.h>

/// Test decomposition algorithm on dense positive definite matrices
template <class DecType> void BM_pos_def(benchmark::State &state) {
  Eigen::Rand::P8_mt19937_64 rng{42};
  typedef typename DecType::MatrixType MatrixType;
  for (auto _ : state) {
    long n = state.range(0);
    state.PauseTiming();
    auto wgen = Eigen::Rand::makeWishartGen(n + 1, MatrixType::Identity(n, n));
    MatrixType a = wgen.generate(rng);
    MatrixType b = Eigen::Rand::normal<MatrixType>(n, 4, rng);
    state.ResumeTiming();

    DecType dec(a);
    dec.solveInPlace(b);
  }
}

/// Test decomposition algorithm on dense positive semi definite matrices
template <class DecType> void BM_pos_sem_def(benchmark::State &state) {
  Eigen::Rand::P8_mt19937_64 rng{42};
  typedef typename DecType::MatrixType MatrixType;
  for (auto _ : state) {
    long n = state.range(0);
    state.PauseTiming();
    MatrixType a_n =
        Eigen::Rand::normal<MatrixType>(n, n - state.range(1), rng);
    MatrixType a = a_n * a_n.transpose();
    MatrixType b = Eigen::Rand::normal<MatrixType>(n, 4, rng);
    state.ResumeTiming();

    DecType dec(a);
    dec.solveInPlace(b);
  }
}

/// Test decomposition algorithm on indefinite matrices
template <class DecType, typename MatrixType = typename DecType::MatrixType>
void BM_indefinite(benchmark::State &state) {
  Eigen::Rand::P8_mt19937_64 rng{42};
  for (auto _ : state) {
    long n = state.range(0);
    state.PauseTiming();
    MatrixType a = sampleGaussianOrthogonalEnsemble(n);
    MatrixType b = Eigen::Rand::normal<MatrixType>(n, 4, rng);
    state.ResumeTiming();

    DecType dec(a);
    dec.solveInPlace(b);
    benchmark::DoNotOptimize(b);
  }
}

/// TODO 1000, 2000 and 3000 take really long time
const std::vector<int64_t> dimArgs = {55, 64, 77, 115, 128, 256, 432,
                                      /*1000, 2000, 4832*/};
const std::vector<int64_t> nonDefArgs = {1, 2, 4, 8, 16, 32};
static void custom_args(benchmark::internal::Benchmark *b) {
  b->ArgName("dim");
  for (auto d : dimArgs)
    b->Arg(d);
  b->Unit(benchmark::kMicrosecond)->UseRealTime()->MinWarmUpTime(0.1);
}
static void custom_args_pos_sem_def(benchmark::internal::Benchmark *b) {
  b->ArgNames({"dim", "non_def"});
  b->ArgsProduct({dimArgs, nonDefArgs});
  b->Unit(benchmark::kMicrosecond)->UseRealTime()->MinWarmUpTime(0.1);
}

BENCHMARK(BM_pos_def<Eigen::LLT<Eigen::MatrixXd>>)->Apply(custom_args);
BENCHMARK(BM_pos_def<Eigen::LDLT<Eigen::MatrixXd>>)->Apply(custom_args);
BENCHMARK(BM_pos_def<Eigen::BunchKaufman<Eigen::MatrixXd>>)->Apply(custom_args);
BENCHMARK(BM_pos_sem_def<Eigen::LLT<Eigen::MatrixXd>>)
    ->Apply(custom_args_pos_sem_def);
BENCHMARK(BM_pos_sem_def<Eigen::LDLT<Eigen::MatrixXd>>)
    ->Apply(custom_args_pos_sem_def);
BENCHMARK(BM_pos_sem_def<Eigen::BunchKaufman<Eigen::MatrixXd>>)
    ->Apply(custom_args_pos_sem_def);
BENCHMARK(BM_indefinite<Eigen::LDLT<Eigen::MatrixXd>>)->Apply(custom_args);
BENCHMARK(BM_indefinite<Eigen::BunchKaufman<Eigen::MatrixXd>>)
    ->Apply(custom_args);
BENCHMARK(BM_indefinite<proxsuite::nlp::linalg::DenseLDLT<double>>)
    ->Apply(custom_args);

BENCHMARK_MAIN();
