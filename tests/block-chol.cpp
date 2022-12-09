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

constexpr isize n = 3;

// clang-format off
BlockKind data[n * n] = {
    BlockKind::Diag,  BlockKind::Zero,  BlockKind::Dense,
    BlockKind::Zero,  BlockKind::Dense, BlockKind::Diag,
    BlockKind::Dense, BlockKind::Diag,  BlockKind::Diag,
};
// clang-format on

isize row_segments[n] = {8, 16, 16};
isize size = []() -> isize {
  isize r = 0;
  for (std::size_t i = 0; i < n; ++i)
    r += row_segments[i];

  return r;
}();

SymbolicBlockMatrix sym_mat{data, row_segments, n, n};

MatrixXs mat = []() -> MatrixXs {
  MatrixXs mat(size, size);
  mat.setZero();

  isize rs0 = row_segments[0];
  isize rs1 = row_segments[1];
  isize rs2 = row_segments[2];
  isize idx0 = 0;
  isize idx1 = rs0;
  isize idx2 = rs0 + rs1;

  mat.block(idx0, idx0, rs0, rs0).diagonal().setRandom();

  mat.block(idx1, idx0, rs1, rs0).setZero();
  mat.block(idx1, idx1, rs1, rs1).setRandom();

  mat.block(idx2, idx0, rs2, rs0).setRandom();
  mat.block(idx2, idx1, rs2, rs1).diagonal().setRandom();
  mat.block(idx2, idx2, rs2, rs2).diagonal().setRandom();
  mat = mat.selfadjointView<Eigen::Lower>();
  return mat;
}();

VectorXs rhs = VectorXs::Random(size);

static void bm_blocked(benchmark::State &s) {
  auto l = mat;
  for (auto _ : s) {
    l = mat;
    BlockLDLT l_block{l, sym_mat};
    l_block.ldlt_in_place();
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

auto l0 = mat;
auto l1 = mat;

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

  isize best_perm[n];
  std::iota(best_perm, best_perm + n, isize(0));

  // dense LDLT

  BlockKind copy_data[n * n];
  isize copy_row_segments[n];
  SymbolicBlockMatrix copy_mat{copy_data, copy_row_segments, n, n};

  BlockLDLT a_block_permuted{l1, copy_mat};
  BlockLDLT a_block_unpermuted{mat, sym_mat};
  a_block_permuted.permute(a_block_unpermuted, best_perm);

  DenseLDLT dense_ldlt(l1);

  MatrixXs reconstr = dense_ldlt.reconstructedMatrix();
  BOOST_CHECK(reconstr.isApprox(mat));

  MatrixXs sol_dense = rhs;
  dense_ldlt.solveInPlace(sol_dense);

  BOOST_CHECK(sol_dense.isApprox(sol_eig));
  BOOST_CHECK(rhs.isApprox(mat * sol_dense));
}

BOOST_FIXTURE_TEST_CASE(test_block_ldlt_ours, ldlt_fixture) {
  fmt::print("Input matrix pattern:\n");
  block_chol::print_sparsity_pattern(sym_mat);

  isize best_perm[n];
  std::iota(best_perm, best_perm + n, isize(0));

  BlockKind copy_data[n * n];
  isize copy_row_segments[n];
  SymbolicBlockMatrix copy_sym_mat{copy_data, copy_row_segments, n, n};

  BlockLDLT block_permuted{l0, copy_sym_mat};
  BlockLDLT block_unpermuted{mat, sym_mat};
  block_permuted.permute(block_unpermuted, best_perm);

  copy_sym_mat.llt_in_place();
  Eigen::ComputationInfo info = block_permuted.ldlt_in_place();
  BOOST_CHECK(info == Eigen::Success);

  sym_mat.llt_in_place();
  fmt::print("Resulting sparsity after pivoting:\n");
  block_chol::print_sparsity_pattern(sym_mat);

  MatrixXs reconstr = block_permuted.reconstructedMatrix();
  BOOST_CHECK(reconstr.isApprox(mat));

  constexpr Scalar prec = 1e-10;

  MatrixXs sol_block = rhs;
  block_permuted.solveInPlace(sol_block);
  Scalar err = (sol_block - sol_eig).lpNorm<-1>();
  fmt::print("err = {:.5e}\n", err);
  BOOST_CHECK(sol_block.isApprox(sol_eig, prec));
  BOOST_CHECK(rhs.isApprox(mat * sol_block, prec));
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
