/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, ",", "\n", "[", "]")

#include "util.hpp"
#include "proxnlp/ldlt-allocator.hpp"

#include <benchmark/benchmark.h>

#include "proxnlp/math.hpp"

using namespace proxnlp;

using linalg::BlockLDLT;
using linalg::DenseLDLT;
using linalg::EigenLDLTWrapper;

constexpr isize n = 3;
constexpr double TOL = 1e-11;
constexpr auto unit = benchmark::kMicrosecond;
const isize ndx = 24;

auto create_problem_structure(isize c1, isize c2, isize c3)
    -> linalg::SymbolicBlockMatrix {
  // clang-format off
  BlockKind *data = new BlockKind[n * n]{
      BlockKind::Diag,  BlockKind::Dense, BlockKind::Dense,
      BlockKind::Dense, BlockKind::Dense, BlockKind::Diag,
      BlockKind::Dense, BlockKind::Diag, BlockKind::Diag
  };
  // clang-format on

  // isize row_segments[n] = {8, 16, 16};
  isize *row_segments = new isize[n]{c1, c2, c3};
  return {data, row_segments, n, n};
}

// isize ncols = ndx;

// struct ldlt_test_fixture {
//   ldlt_test_fixture() : mat(), rhs(), ldlt() { this->init(); }
//   ~ldlt_test_fixture() = default;

//   MatrixXs mat;
//   MatrixXs rhs;
//   Eigen::LDLT<MatrixXs> ldlt;
//   MatrixXs sol_eig;
//   isize size;

//   void init() {
//     mat = getRandomSymmetricBlockMatrix(sym_mat);
//     ldlt.compute(mat);
//     size = mat.cols();
//     rhs = MatrixXs::Random(size, ncols);
//     sol_eig = ldlt.solve(rhs);
//   }
// };

// struct ldlt_bench_fixture : benchmark::Fixture, ldlt_test_fixture {
//   void SetUp(const benchmark::State &) override { this->init(); }
//   void TearDown(const benchmark::State &) override {}
// };

// BENCHMARK_DEFINE_F(ldlt_bench_fixture, block_sparse)(benchmark::State &s) {
//   BlockLDLT<Scalar> block_ldlt(size, sym_mat);
//   block_ldlt.findSparsifyingPermutation();
//   auto b = rhs;
//   for (auto _ : s) {
//     b = rhs;
//     block_ldlt.compute(mat);
//     block_ldlt.solveInPlace(b);
//     if (block_ldlt.info() != Eigen::Success) {
//       s.SkipWithError("BlockLDLT computation failed.");
//       break;
//     }
//   }
// }

static void ldlt_recursive(benchmark::State &s) {
  const isize c1_size = static_cast<isize>(s.range(0));
  const isize c2c3_size = c1_size * 2;
  const isize matrix_size = c1_size + c2c3_size * 2;
  VectorXs rhs = VectorXs::Random(matrix_size);
  VectorXs b(matrix_size);
  DenseLDLT<Scalar> dense_ldlt(matrix_size);
  // Construct the matrix
  auto sym_mat = create_problem_structure(c1_size, c2c3_size, c2c3_size);
  MatrixXs mat = getRandomSymmetricBlockMatrix(sym_mat);

  for (auto _ : s) {
    s.PauseTiming();
    b = rhs;
    s.ResumeTiming();
    dense_ldlt.compute(mat);
    dense_ldlt.solveInPlace(rhs);
    s.PauseTiming();
    if (dense_ldlt.info() != Eigen::Success) {
      s.SkipWithError("DenseLDLT computation failed.");
      break;
    }
  }
}

// BENCHMARK_DEFINE_F(ldlt_bench_fixture, bunchkaufman)(benchmark::State &s) {
//   Eigen::BunchKaufman<MatrixXs> lblt(size);
//   auto b = rhs;
//   for (auto _ : s) {
//     lblt.compute(mat);
//     b = lblt.solve(rhs);
//     benchmark::DoNotOptimize(lblt);
//   }
// }

// BENCHMARK_DEFINE_F(ldlt_bench_fixture, eigen_ldlt)(benchmark::State &s) {
//   Eigen::LDLT<MatrixXs> ldlt(size);
//   auto b = rhs;
//   for (auto _ : s) {
//     b = rhs;
//     ldlt.compute(mat);
//     ldlt.solveInPlace(b);
//     if (ldlt.info() != Eigen::Success) {
//       s.SkipWithError("Eigen::LDLT computation failed.");
//       break;
//     }
//     benchmark::DoNotOptimize(ldlt);
//   }
// }

BENCHMARK(ldlt_recursive)
    ->RangeMultiplier(2)
    ->Range(4, 512)
    ->Unit(unit)
    ->MinTime(2.)
    ->MinWarmUpTime(0.1);
// BENCHMARK_REGISTER_F(ldlt_bench_fixture, block_sparse)->Unit(unit);
// BENCHMARK_REGISTER_F(ldlt_bench_fixture, eigen_ldlt)->Unit(unit);
// BENCHMARK_REGISTER_F(ldlt_bench_fixture, bunchkaufman)->Unit(unit);

// #ifdef PROXNLP_ENABLE_PROXSUITE_LDLT

// BENCHMARK_DEFINE_F(ldlt_bench_fixture, proxsuite_ldlt)(benchmark::State &s) {

//   linalg::ProxSuiteLDLTWrapper<Scalar> ps_ldlt(mat.rows(), rhs.cols() + 1);
//   auto sol_ps = rhs;
//   for (auto _ : s) {
//     sol_ps = rhs;
//     ps_ldlt.compute(mat);
//     ps_ldlt.solveInPlace(sol_ps);
//   }
// }
// BENCHMARK_REGISTER_F(ldlt_bench_fixture, proxsuite_ldlt)->Unit(unit);

// #endif

BENCHMARK_MAIN();
