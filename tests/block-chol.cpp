#include "proxnlp/blocks.hpp"

#include <benchmark/benchmark.h>

#include <fmt/core.h>

using namespace proxnlp;
using block_chol::BlockKind;
using block_chol::BlockMatrix;
using block_chol::isize;
using block_chol::MatrixRef;
using block_chol::Scalar;
using block_chol::SymbolicBlockMatrix;

using MatrixXs = MatrixRef::PlainMatrix;
using VectorXs = math_types<Scalar>::VectorXs;

constexpr isize n = 3;

// clang-format off
BlockKind data[n * n] = {
    BlockKind::Diag,  BlockKind::Zero,  BlockKind::Dense,
    BlockKind::Zero,  BlockKind::Dense, BlockKind::Diag,
    BlockKind::Dense, BlockKind::Diag,  BlockKind::Diag,
};
// clang-format on

isize row_segments[n] = {16, 32, 32};

SymbolicBlockMatrix mat{{data, row_segments, n, n}};

auto a = []() -> MatrixXs {
  MatrixXs mat(80, 80);
  mat.setZero();

  mat.block(0, 0, 16, 16).diagonal().setRandom();

  mat.block(16, 0, 32, 32).setZero();
  mat.block(16, 16, 32, 32).setRandom();

  mat.block(48, 0, 32, 32).setRandom();
  mat.block(48, 16, 32, 32).diagonal().setRandom();
  mat.block(48, 48, 32, 32).diagonal().setRandom();
  return mat;
}();

VectorXs rhs = VectorXs::Random(80);

void bm_blocked(benchmark::State &s) {
  auto l = a;
  for (auto _ : s) {
    l = a;
    BlockMatrix l_block{MatrixRef(l), mat};
    l_block.ldlt_in_place();
  }
}

void bm_unblocked(benchmark::State &s) {
  auto l = a;
  for (auto _ : s) {
    l = a;
    block_chol::backend::ldlt_in_place(MatrixRef(l));
    l.template triangularView<Eigen::StrictlyUpper>().setZero();
  }
}

auto unit = benchmark::kMicrosecond;
BENCHMARK(bm_unblocked)->Unit(unit);
BENCHMARK(bm_blocked)->Unit(unit);

int main(int argc, char **argv) {

  fmt::print("Input matrix pattern:\n");
  mat.dump();

  isize best_perm[n];

  {
    BlockKind copy_data[n * n];
    isize copy_row_segments[n];
    isize work[n];
    SymbolicBlockMatrix copy_mat{{copy_data, copy_row_segments, n, n}};
    copy_mat.brute_force_best_permutation(mat, best_perm, work);
  }

  auto l0 = a;
  {
    BlockKind copy_data[n * n];
    isize copy_row_segments[n];
    SymbolicBlockMatrix copy_mat{{copy_data, copy_row_segments, n, n}};

    BlockMatrix a_block_permuted{l0, copy_mat};
    BlockMatrix a_block_unpermuted{a, mat};
    a_block_permuted.permute(a_block_unpermuted, best_perm);

    copy_mat.llt_in_place();
    a_block_permuted.ldlt_in_place();
    l0.template triangularView<Eigen::StrictlyUpper>().setZero();
  }

  auto l1 = a;
  {
    BlockKind copy_data[n * n];
    isize copy_row_segments[n];
    SymbolicBlockMatrix copy_mat{{copy_data, copy_row_segments, n, n}};

    BlockMatrix a_block_permuted{l1, copy_mat};
    BlockMatrix a_block_unpermuted{a, mat};
    a_block_permuted.permute(a_block_unpermuted, best_perm);

    block_chol::backend::ldlt_in_place(l1);
    l1.template triangularView<Eigen::StrictlyUpper>().setZero();
  }

  fmt::print("Err: {:g}\n", (l0 - l1).norm());

  mat.llt_in_place();
  mat.dump();

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
