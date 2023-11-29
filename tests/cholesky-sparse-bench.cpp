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

namespace {
constexpr isize n = 3;
constexpr auto unit = benchmark::kMicrosecond;

auto create_problem_structure(isize c1, isize c2, isize c3)
    -> linalg::SymbolicBlockMatrix {
  // clang-format off
  BlockKind *data = new BlockKind[n * n]{
      BlockKind::Diag,  BlockKind::Dense, BlockKind::Dense,
      BlockKind::Dense, BlockKind::Dense, BlockKind::Diag,
      BlockKind::Dense, BlockKind::Diag, BlockKind::Diag
  };
  // clang-format on

  isize *row_segments = new isize[n]{c1, c2, c3};
  return {data, row_segments, n, n};
}

/// Return true on solver.compute success
template <typename LDLT> bool success(const LDLT &ldlt) {
  return ldlt.info() == Eigen::Success;
}

/// Eigen::BunchKaufman::info is not implemented
bool success(const Eigen::BunchKaufman<MatrixXs> & /* ldlt */) { return true; }

/// Construct a standard LDLT problem
template <typename LDLT>
LDLT construct(isize matrix_size, const SymbolicBlockMatrix & /* sym_mat */) {
  return LDLT(matrix_size);
}

/// BlockLDLT need SymbolicBlockMatrix as argument
template <>
BlockLDLT<Scalar>
construct<BlockLDLT<Scalar>>(isize matrix_size,
                             const SymbolicBlockMatrix &sym_mat) {
  BlockLDLT<Scalar> b(matrix_size, sym_mat);
  b.findSparsifyingPermutation();
  return b;
}

/// Store all problem variables
/// TODO create_problem_structure must be able to return different matrix
/// structure
template <typename LDLT> struct Problem {
  Problem(int64_t c1_size_)
      : c1_size(static_cast<isize>(c1_size_)), c2c3_size(c1_size * 2),
        matrix_size(c1_size + c2c3_size * 2),
        sym_mat(create_problem_structure(c1_size, c2c3_size, c2c3_size)),
        mat(getRandomSymmetricBlockMatrix(sym_mat)),
        rhs(VectorXs::Random(matrix_size)),
        ldlt(construct<LDLT>(matrix_size, sym_mat)) {}

  const isize c1_size;
  const isize c2c3_size;
  const isize matrix_size;
  const SymbolicBlockMatrix sym_mat;
  const MatrixXs mat;
  const VectorXs rhs;
  LDLT ldlt;
};

/// Benchmark LDLT factorization (compute)
template <typename LDLT> void ldlt_compute(benchmark::State &s) {
  Problem<LDLT> p(s.range(0));

  for (auto _ : s) {
    p.ldlt.compute(p.mat);
    if (!success(p.ldlt)) {
      s.SkipWithError("computation failed.");
      break;
    }
  }
}

/// Benchmark solveInPlace
template <typename LDLT> void ldlt_solve_in_place(benchmark::State &s) {
  Problem<LDLT> p(s.range(0));
  VectorXs b(p.matrix_size);

  p.ldlt.compute(p.mat);
  if (!success(p.ldlt)) {
    s.SkipWithError("computation failed.");
    return;
  }

  for (auto _ : s) {
    s.PauseTiming();
    b = p.rhs;
    s.ResumeTiming();
    p.ldlt.solveInPlace(b);
  }
}

/// Benchmark solve
template <typename LDLT> void ldlt_solve(benchmark::State &s) {
  Problem<LDLT> p(s.range(0));

  p.ldlt.compute(p.mat);
  if (!success(p.ldlt)) {
    s.SkipWithError("computation failed.");
    return;
  }

  for (auto _ : s) {
    VectorXs x(p.ldlt.solve(p.rhs));
  }
}

void default_arguments(benchmark::internal::Benchmark *b) {
  b->RangeMultiplier(2)->Range(4, 512)->Unit(unit)->MinTime(2.)->MinWarmUpTime(
      0.1);
}

} // namespace

BENCHMARK(ldlt_compute<DenseLDLT<Scalar>>)->Apply(default_arguments);
BENCHMARK(ldlt_compute<Eigen::LDLT<MatrixXs>>)->Apply(default_arguments);
BENCHMARK(ldlt_compute<Eigen::BunchKaufman<MatrixXs>>)
    ->Apply(default_arguments);
BENCHMARK(ldlt_compute<BlockLDLT<Scalar>>)->Apply(default_arguments);
BENCHMARK(ldlt_solve_in_place<DenseLDLT<Scalar>>)->Apply(default_arguments);
BENCHMARK(ldlt_solve_in_place<Eigen::LDLT<MatrixXs>>)->Apply(default_arguments);
BENCHMARK(ldlt_solve<Eigen::BunchKaufman<MatrixXs>>)->Apply(default_arguments);
BENCHMARK(ldlt_solve_in_place<BlockLDLT<Scalar>>)->Apply(default_arguments);

#ifdef PROXNLP_ENABLE_PROXSUITE_LDLT

BENCHMARK(ldlt_compute<linalg::ProxSuiteLDLTWrapper<Scalar>>)
    ->Apply(default_arguments);
BENCHMARK(ldlt_solve_in_place<linalg::ProxSuiteLDLTWrapper<Scalar>>)
    ->Apply(default_arguments);

#endif

BENCHMARK_MAIN();
