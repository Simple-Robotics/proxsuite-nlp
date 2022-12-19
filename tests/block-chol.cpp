#include "proxnlp/blocks.hpp"

#include <benchmark/benchmark.h>
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include <fmt/ostream.h>

namespace utf = boost::unit_test;

using namespace proxnlp;
using block_chol::BlockKind;
using block_chol::BlockLDLT;
using block_chol::DenseLDLT;
using block_chol::isize;
using block_chol::MatrixRef;
using block_chol::Scalar;
using block_chol::SymbolicBlockMatrix;

using MatrixXs = MatrixRef::PlainMatrix;
using VectorXs = math_types<Scalar>::VectorXs;

MatrixXs get_block_matrix(SymbolicBlockMatrix const &sym) {
  isize *row_segments = sym.segment_lens;
  auto n = std::size_t(sym.nsegments());
  isize size = 0;
  for (std::size_t i = 0; i < n; ++i)
    size += row_segments[i];

  MatrixXs mat(size, size);
  mat.setZero();

  isize startRow = 0;
  isize startCol = 0;
  for (unsigned i = 0; i < n; ++i) {
    isize blockRows = row_segments[i];
    startCol = 0;
    for (unsigned j = 0; j <= i; ++j) {
      isize blockCols = row_segments[j];
      const BlockKind kind = sym(i, j);
      fmt::print("Filling in block ({}, {}, {}, {}), kind = {}\n", startRow,
                 startCol, blockRows, blockCols, kind);
      auto block = mat.block(startRow, startCol, blockRows, blockCols);
      switch (kind) {
      case BlockKind::Zero:
        block.setZero();
        break;
      case BlockKind::Dense:
        block.diagonal(0).setConstant(2.);
        block.diagonal(1).setConstant(-1.);
        block.diagonal(-1).setConstant(-1.);
        break;
      case BlockKind::Diag:
        block.diagonal().setRandom();
      default:
        break;
      }
      startCol += blockCols;
    }
    startRow += blockRows;
  }
  mat = mat.selfadjointView<Eigen::Lower>();
  return mat;
}

constexpr isize n = 2;

// clang-format off
BlockKind data[n * n] = {
    BlockKind::Diag,  BlockKind::Dense,
    BlockKind::Dense, BlockKind::Diag,
};
// clang-format on

// isize row_segments[n] = {8, 16, 16};
isize row_segments[n] = {4, 8};
SymbolicBlockMatrix sym_mat{data, row_segments, n, n};
MatrixXs mat = get_block_matrix(sym_mat);
isize size = mat.cols();

VectorXs rhs = VectorXs::Random(size);

static void bm_blocked(benchmark::State &s) {
  auto l = mat;
  for (auto _ : s) {
    l = mat;
    BlockLDLT l_block{l, sym_mat};
    l_block.compute(l);
  }
}

static void bm_unblocked(benchmark::State &s) {
  for (auto _ : s) {
    DenseLDLT dense_ldlt(mat);
  }
}

static void bm_eigen_ldlt(benchmark::State &s) {
  for (auto _ : s) {
    Eigen::LDLT<MatrixXs> ldlt(mat);
    benchmark::DoNotOptimize(ldlt);
  }
}

auto unit = benchmark::kMicrosecond;
BENCHMARK(bm_unblocked)->Unit(unit);
BENCHMARK(bm_blocked)->Unit(unit);
BENCHMARK(bm_eigen_ldlt)->Unit(unit);

struct ldlt_fixture {
  ldlt_fixture() : ldlt(mat) { sol_eig = ldlt.solve(rhs); }
  ~ldlt_fixture() = default;

  Eigen::LDLT<MatrixXs> ldlt;
  MatrixXs sol_eig;
};

BOOST_FIXTURE_TEST_CASE(test_eigen_ldlt, ldlt_fixture) {
  MatrixXs reconstr = ldlt.reconstructedMatrix();
  BOOST_CHECK(reconstr.isApprox(mat));
  BOOST_CHECK(rhs.isApprox(mat * sol_eig));
}

BOOST_FIXTURE_TEST_CASE(test_dense_ldlt_ours, ldlt_fixture) {
  // dense LDLT
  DenseLDLT dense_ldlt(mat);

  MatrixXs reconstr = dense_ldlt.reconstructedMatrix();
  BOOST_CHECK(reconstr.isApprox(mat));

  MatrixXs sol_dense = rhs;
  dense_ldlt.solveInPlace(sol_dense);

  Scalar dense_err = math::infty_norm(sol_dense - sol_eig);
  fmt::print("Dense err = {:.5e}\n", dense_err);
  BOOST_CHECK(sol_dense.isApprox(sol_eig));
  BOOST_CHECK(rhs.isApprox(mat * sol_dense));
}

BOOST_FIXTURE_TEST_CASE(test_block_ldlt_ours, ldlt_fixture) {
  fmt::print("Input matrix pattern:\n");
  block_chol::print_sparsity_pattern(sym_mat);

  auto l0 = mat;
  auto l1 = mat;
  BlockLDLT block_permuted(l0, sym_mat);
  BlockLDLT block_unpermuted(l1, sym_mat);
  std::vector<isize> best_perm(n);
  std::iota(best_perm.begin(), best_perm.end(), isize(0));
  block_permuted.performAnalysis();
  // block_permuted.permute(block_unpermuted, best_perm.data());

  fmt::print("block_permuted.structure():\n");
  block_chol::print_sparsity_pattern(block_permuted.structure());


  block_permuted.compute(l1);
  Eigen::ComputationInfo info = block_permuted.info();
  BOOST_REQUIRE(info == Eigen::Success);

  sym_mat.llt_in_place();
  fmt::print("Unpermuted LLT :\n");
  block_chol::print_sparsity_pattern(sym_mat);

  MatrixXs reconstr = block_permuted.reconstructedMatrix();
  BOOST_CHECK(reconstr.isApprox(mat));
  auto reconstr_err = math::infty_norm(reconstr - mat);
  fmt::print("Block reconstr err = {:.5e}\n", reconstr_err);

  MatrixXs sol_block = rhs;
  block_permuted.solveInPlace(sol_block);
  Scalar err = math::infty_norm(sol_block - sol_eig);
  fmt::print("err = {:.5e}\n", err);
  BOOST_CHECK(sol_block.isApprox(sol_eig));
  BOOST_CHECK(rhs.isApprox(mat * sol_block));
}

int main(int argc, char **argv) {
  // call default test initialization function
  // see Boost.Test docs:
  // https://www.boost.org/doc/libs/1_80_0/libs/test/doc/html/boost_test/adv_scenarios/shared_lib_customizations/entry_point.html
  int tests_result = utf::unit_test_main(&init_unit_test, argc, argv);

  benchmark::Initialize(&argc, argv);
  // run benchmarks
  benchmark::RunSpecifiedBenchmarks();

  return tests_result;
}
