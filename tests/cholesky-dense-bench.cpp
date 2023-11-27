/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#include "proxnlp/linalg/bunchkaufman.hpp"
#include "proxnlp/linalg/dense.hpp"
#include "util.hpp"
#include <benchmark/benchmark.h>

template <class DecType> void BM_posdef(benchmark::State &state) {
  Eigen::Rand::P8_mt19937_64 rng{42};
  typedef typename DecType::MatrixType MatrixType;
  for (auto _ : state) {
    long n = state.range();
    state.PauseTiming();
    auto wgen = Eigen::Rand::makeWishartGen(n + 1, MatrixType::Identity(n, n));
    MatrixType a = wgen.generate(rng);
    MatrixType b = Eigen::Rand::normal<MatrixType>(n, 4, rng);
    state.ResumeTiming();

    DecType dec(a);
    dec.solveInPlace(b);
  }
}

template <class DecType, typename MatrixType = typename DecType::MatrixType>
void BM_indefinite(benchmark::State &state) {
  Eigen::Rand::P8_mt19937_64 rng{42};
  for (auto _ : state) {
    long n = state.range();
    state.PauseTiming();
    MatrixType a = sampleGaussianOrhogonalEnsemble(n);
    MatrixType b = Eigen::Rand::normal<MatrixType>(n, 4, rng);
    state.ResumeTiming();

    DecType dec(a);
    dec.solveInPlace(b);
    benchmark::DoNotOptimize(b);
  }
}

const std::vector<long> dimArgs = {55,  64,  77,   115,  128,
                                   256, 432, 1000, 2000, 4832};
static void custom_args(benchmark::internal::Benchmark *b) {

  b->ArgName("dim");
  for (auto d : dimArgs)
    b->Arg(d);
  b->Unit(benchmark::kMicrosecond)->Repetitions(5)->UseRealTime();
}

BENCHMARK_TEMPLATE(BM_posdef, Eigen::LLT<Eigen::MatrixXd>)->Apply(custom_args);
BENCHMARK_TEMPLATE(BM_posdef, Eigen::LDLT<Eigen::MatrixXd>)->Apply(custom_args);
BENCHMARK_TEMPLATE(BM_posdef, Eigen::BunchKaufman<Eigen::MatrixXd>)
    ->Apply(custom_args);
BENCHMARK_TEMPLATE(BM_indefinite, Eigen::LDLT<Eigen::MatrixXd>)
    ->Apply(custom_args);
BENCHMARK_TEMPLATE(BM_indefinite, Eigen::BunchKaufman<Eigen::MatrixXd>)
    ->Apply(custom_args);
BENCHMARK_TEMPLATE(BM_indefinite, proxnlp::linalg::DenseLDLT<double>)
    ->Apply(custom_args);

BENCHMARK_MAIN();
