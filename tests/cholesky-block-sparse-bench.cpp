/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, ",", "\n", "[", "]")

#include "util.hpp"
#include "proxsuite-nlp/ldlt-allocator.hpp"

#include <benchmark/benchmark.h>

#include "proxsuite-nlp/math.hpp"

using namespace proxsuite::nlp;

using linalg::BlockLDLT;
using linalg::DenseLDLT;

#ifdef PROXSUITE_NLP_USE_PROXSUITE_LDLT

using linalg::ProxSuiteLDLTWrapper;

#endif

namespace {
constexpr auto unit = benchmark::kMicrosecond;

auto create_problem_structure(isize matrix_structure_type, isize pb_size)
    -> linalg::SymbolicBlockMatrix {

  BlockKind *data = nullptr;
  isize *row_segments = nullptr;
  isize nb_blocks = 0;

  switch (matrix_structure_type) {
  case 0:
    nb_blocks = 3;
    // clang-format off
    data = new BlockKind[static_cast<ulong>(nb_blocks * nb_blocks)]{
      BlockKind::Diag,  BlockKind::Dense, BlockKind::Dense,
      BlockKind::Dense, BlockKind::Dense, BlockKind::Diag,
      BlockKind::Dense, BlockKind::Diag, BlockKind::Diag
    };
    // clang-format on
    row_segments = new isize[static_cast<ulong>(nb_blocks)]{
        pb_size, pb_size * 2, pb_size * 2};
    break;
  case 1:
    nb_blocks = 3;
    // clang-format off
    data = new BlockKind[static_cast<ulong>(nb_blocks * nb_blocks)]{
      BlockKind::Diag,  BlockKind::Zero, BlockKind::Dense,
      BlockKind::Dense, BlockKind::Diag, BlockKind::Zero,
      BlockKind::Zero, BlockKind::Dense, BlockKind::Diag
    };
    // clang-format on
    row_segments = new isize[static_cast<ulong>(nb_blocks)]{
        pb_size, pb_size * 2, pb_size * 2};
    break;
  default:
    throw std::runtime_error("Unknonw matrix_structure_type");
  }

  return {data, row_segments, nb_blocks, nb_blocks};
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

#ifdef PROXSUITE_NLP_USE_PROXSUITE_LDLT

template <>
ProxSuiteLDLTWrapper<Scalar>
construct<ProxSuiteLDLTWrapper<Scalar>>(isize matrix_size,
                                        const SymbolicBlockMatrix &) {
  return ProxSuiteLDLTWrapper<Scalar>(matrix_size, matrix_size);
}

#endif

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
template <typename LDLT> struct Problem {
  Problem(int64_t pb_size_, int64_t matrix_structure_type_)
      : pb_size(static_cast<isize>(pb_size_)),
        sym_mat(
            create_problem_structure(static_cast<isize>(matrix_structure_type_),
                                     static_cast<isize>(pb_size_))),
        mat(getRandomSymmetricBlockMatrix(sym_mat)), matrix_size(mat.cols()),
        rhs(VectorXs::Random(matrix_size)),
        ldlt(construct<LDLT>(matrix_size, sym_mat)) {}

  const isize pb_size;
  const SymbolicBlockMatrix sym_mat;
  const MatrixXs mat;
  const isize matrix_size;
  const VectorXs rhs;
  LDLT ldlt;
};

/// Benchmark LDLT factorization (compute)
template <typename LDLT> void ldlt_compute(benchmark::State &s) {
  Problem<LDLT> p(s.range(0), s.range(1));

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
  Problem<LDLT> p(s.range(0), s.range(1));
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
  Problem<LDLT> p(s.range(0), s.range(1));

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
  b->Unit(unit)->MinWarmUpTime(0.1);
  b->ArgsProduct({{4, 8, 16, 32, 64, 128, 256, 512}, {0, 1}});
}
void default_arguments_compute(benchmark::internal::Benchmark *b) {
  default_arguments(b);
  b->MinTime(2.);
}

} // namespace

BENCHMARK_TEMPLATE(ldlt_compute, DenseLDLT<Scalar>)
    ->Apply(default_arguments_compute);
BENCHMARK_TEMPLATE(ldlt_compute, Eigen::LDLT<MatrixXs>)
    ->Apply(default_arguments_compute);
BENCHMARK_TEMPLATE(ldlt_compute, Eigen::BunchKaufman<MatrixXs>)
    ->Apply(default_arguments_compute);
BENCHMARK_TEMPLATE(ldlt_compute, BlockLDLT<Scalar>)
    ->Apply(default_arguments_compute);
BENCHMARK_TEMPLATE(ldlt_solve_in_place, DenseLDLT<Scalar>)
    ->Apply(default_arguments);
BENCHMARK_TEMPLATE(ldlt_solve_in_place, Eigen::LDLT<MatrixXs>)
    ->Apply(default_arguments);
BENCHMARK_TEMPLATE(ldlt_solve, Eigen::BunchKaufman<MatrixXs>)
    ->Apply(default_arguments);
BENCHMARK_TEMPLATE(ldlt_solve_in_place, BlockLDLT<Scalar>)
    ->Apply(default_arguments);

#ifdef PROXSUITE_NLP_USE_PROXSUITE_LDLT

BENCHMARK_TEMPLATE(ldlt_compute, ProxSuiteLDLTWrapper<Scalar>)
    ->Apply(default_arguments_compute);
BENCHMARK_TEMPLATE(ldlt_solve_in_place, ProxSuiteLDLTWrapper<Scalar>)
    ->Apply(default_arguments);

#endif

BENCHMARK_MAIN();
