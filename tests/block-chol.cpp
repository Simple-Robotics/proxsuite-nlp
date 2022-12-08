#include "proxnlp/blocks.hpp"

#include <benchmark/benchmark.h>
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include <fmt/ostream.h>

namespace utf = boost::unit_test;

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

isize row_segments[n] = {4, 8, 8};
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
    BlockMatrix l_block{l, sym_mat};
    l_block.ldlt_in_place();
  }
}

static void bm_unblocked(benchmark::State &s) {
  auto l = mat;
  for (auto _ : s) {
    l = mat;
    block_chol::backend::dense_ldlt_in_place(l);
    benchmark::DoNotOptimize(l);
  }
}

static void bm_eigen_ldlt(benchmark::State &s) {
  auto l = mat;
  for (auto _ : s) {
    l = mat;
    Eigen::LDLT<MatrixXs> ldlt(l);
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
  BOOST_CHECK((mat * sol_eig).isApprox(rhs));
}

BOOST_FIXTURE_TEST_CASE(test_dense_ldlt_ours, ldlt_fixture) {

  isize best_perm[n];
  std::iota(best_perm, best_perm + n, isize(0));

  // dense LDLT

  BlockKind copy_data[n * n];
  isize copy_row_segments[n];
  SymbolicBlockMatrix copy_mat{copy_data, copy_row_segments, n, n};

  BlockMatrix a_block_permuted{l1, copy_mat};
  BlockMatrix a_block_unpermuted{mat, sym_mat};
  a_block_permuted.permute(a_block_unpermuted, best_perm);

  block_chol::backend::dense_ldlt_in_place(l1);

  MatrixXs reconstruct_l1 = block_chol::backend::dense_ldlt_reconstruct(l1);
  BOOST_CHECK(mat.isApprox(reconstruct_l1));

  // now perform solve
  MatrixXs sol1 = rhs;
  block_chol::backend::dense_ldlt_solve_in_place(l1, sol1);

  BOOST_CHECK(sol1.isApprox(sol_eig));
  BOOST_CHECK(rhs.isApprox(mat * sol1));
}

BOOST_FIXTURE_TEST_CASE(test_block_ldlt_ours, ldlt_fixture) {
  fmt::print("Input matrix pattern:\n");
  block_chol::print_sparsity_pattern(sym_mat);

  isize best_perm[n];
  std::iota(best_perm, best_perm + n, isize(0));

  BlockKind copy_data[n * n];
  isize copy_row_segments[n];
  SymbolicBlockMatrix copy_sym_mat{copy_data, copy_row_segments, n, n};

  BlockMatrix a_block_permuted{l0, copy_sym_mat};
  BlockMatrix a_block_unpermuted{mat, sym_mat};
  a_block_permuted.permute(a_block_unpermuted, best_perm);

  copy_sym_mat.llt_in_place();
  Eigen::ComputationInfo info = a_block_permuted.ldlt_in_place();
  BOOST_CHECK(info == Eigen::Success);

  sym_mat.llt_in_place();
  fmt::print("Resulting sparsity after pivoting:\n");
  block_chol::print_sparsity_pattern(sym_mat);
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
