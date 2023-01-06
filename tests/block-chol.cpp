/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, ",", "\n", "[", "]")

#include "block-test.hpp"

#include <benchmark/benchmark.h>
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include <fmt/ostream.h>
#include <fmt/ranges.h>

namespace utf = boost::unit_test;

using linalg::BlockLDLT;
using linalg::DenseLDLT;

constexpr isize n = 3;

// clang-format off
BlockKind data[n * n] = {
    BlockKind::Dense,  BlockKind::Dense, BlockKind::Zero,
    BlockKind::Dense, BlockKind::Diag, BlockKind::Zero,
    BlockKind::Zero, BlockKind::Zero, BlockKind::Diag
};
// clang-format on

// isize row_segments[n] = {8, 16, 16};
isize row_segments[n] = {24, 30, 10};
SymbolicBlockMatrix sym_mat{data, row_segments, n, n};
MatrixXs mat = get_block_matrix(sym_mat);
isize size = mat.cols();

VectorXs rhs = VectorXs::Random(size);

static void bm_blocked(benchmark::State &s) {
  BlockLDLT<Scalar> block_ldlt(size, sym_mat);
  block_ldlt.findSparsifyingPermutation();
  block_ldlt.updateBlockPermutationMatrix(sym_mat);
  for (auto _ : s) {
    block_ldlt.compute(mat);
    auto b = rhs;
    block_ldlt.solveInPlace(b);
    if (block_ldlt.info() != Eigen::Success) {
      s.SkipWithError("BlockLDLT computation failed.");
      break;
    }
    if (!rhs.isApprox(mat * b)) {
      s.SkipWithError("BlockLDLT gave wrong solution.");
      break;
    }
  }
}

static void bm_unblocked(benchmark::State &s) {
  DenseLDLT<Scalar> dense_ldlt(size);
  for (auto _ : s) {
    dense_ldlt.compute(mat);
    auto b = rhs;
    dense_ldlt.solveInPlace(b);
    if (dense_ldlt.info() != Eigen::Success) {
      s.SkipWithError("DenseLDLT computation failed.");
      break;
    }
    if (!rhs.isApprox(mat * b)) {
      Scalar err = math::infty_norm(rhs - (mat * b));
      s.SkipWithError(
          fmt::format("DenseLDLT gave wrong solution, err={:.4e}.", err)
              .c_str());
      break;
    }
  }
}

static void bm_eigen_ldlt(benchmark::State &s) {
  Eigen::LDLT<MatrixXs> ldlt(size);
  for (auto _ : s) {
    ldlt.compute(mat);
    auto b = rhs;
    ldlt.solveInPlace(b);
    if (ldlt.info() != Eigen::Success) {
      s.SkipWithError("Eigen::LDLT computation failed.");
      break;
    }
    if (!rhs.isApprox(mat * b)) {
      Scalar err = math::infty_norm(rhs - (mat * b));
      s.SkipWithError(
          fmt::format("Eigen::LDLT gave wrong solution, err={:.4e}.", err)
              .c_str());
      break;
    }
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
  BOOST_REQUIRE(ldlt.info() == Eigen::Success);
  BOOST_CHECK(reconstr.isApprox(mat));
  BOOST_CHECK(rhs.isApprox(mat * sol_eig));
}

BOOST_FIXTURE_TEST_CASE(test_eigen_ldlt_wrap, ldlt_fixture) {
  linalg::EigenLDLTWrapper<Scalar> ldlt_wrap(mat);
  BOOST_REQUIRE(ldlt_wrap.info() == Eigen::Success);

  MatrixXs reconstr = ldlt_wrap.reconstructedMatrix();
  BOOST_CHECK(reconstr.isApprox(mat));

  MatrixXs sol_wrap = rhs;
  ldlt_wrap.solveInPlace(sol_wrap);

  BOOST_CHECK(sol_wrap.isApprox(sol_eig));
  BOOST_CHECK(ldlt_wrap.matrixLDLT().isApprox(ldlt.matrixLDLT()));
}

BOOST_FIXTURE_TEST_CASE(test_dense_ldlt_ours, ldlt_fixture) {
  // dense LDLT
  DenseLDLT<Scalar> dense_ldlt(mat);
  BOOST_REQUIRE(dense_ldlt.info() == Eigen::Success);

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
  linalg::print_sparsity_pattern(sym_mat);
  BOOST_REQUIRE(sym_mat.check_if_symmetric());

  BlockLDLT<Scalar> block_permuted(size, sym_mat);
  block_permuted.findSparsifyingPermutation();
  block_permuted.updateBlockPermutationMatrix(sym_mat);
  block_permuted.compute(mat);
  auto best_perm = block_permuted.blockPermIndices();
  fmt::print("Optimal permutation: {}\n",
             fmt::join(best_perm, best_perm + n, ", "));

  {
    auto copy_sym = sym_mat.copy();
    linalg::symbolic_deep_copy(sym_mat, copy_sym, best_perm);
    fmt::print("Permuted structure:\n");
    linalg::print_sparsity_pattern(copy_sym);
  }

  fmt::print("Optimized structure (nnz={:d}):\n",
             block_permuted.structure().count_nnz());
  linalg::print_sparsity_pattern(block_permuted.structure());

  auto pmat = block_permuted.permutationP();
  fmt::print("Permutation matrix: {}\n", pmat.indices().transpose());

  Eigen::ComputationInfo info = block_permuted.info();
  BOOST_REQUIRE(info == Eigen::Success);

  {
    auto copy_sym_mat = sym_mat.copy();
    copy_sym_mat.llt_in_place();
    fmt::print("Un-permuted (suboptimal) LLT (nnz={:d}):\n",
               copy_sym_mat.count_nnz());
    linalg::print_sparsity_pattern(copy_sym_mat);
  }

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
