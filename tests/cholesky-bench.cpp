/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, ",", "\n", "[", "]")

#include "util.hpp"
#include "proxnlp/ldlt-allocator.hpp"

#include <benchmark/benchmark.h>
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include "proxnlp/math.hpp"

namespace utf = boost::unit_test;
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

// BOOST_FIXTURE_TEST_CASE(test_eigen_ldlt, ldlt_test_fixture,
//                         *utf::tolerance(TOL)) {
//   MatrixXs reconstr = ldlt.reconstructedMatrix();
//   BOOST_REQUIRE(ldlt.info() == Eigen::Success);
//   BOOST_CHECK(reconstr.isApprox(mat, TOL));
//   BOOST_CHECK(rhs.isApprox(mat * sol_eig, TOL));
// }

// BOOST_FIXTURE_TEST_CASE(test_eigen_ldlt_wrap, ldlt_test_fixture,
//                         *utf::tolerance(TOL)) {
//   EigenLDLTWrapper<Scalar> ldlt_wrap(mat);
//   BOOST_REQUIRE(ldlt_wrap.info() == Eigen::Success);

//   MatrixXs reconstr = ldlt_wrap.reconstructedMatrix();
//   BOOST_CHECK(reconstr.isApprox(mat));

//   MatrixXs sol_wrap = rhs;
//   ldlt_wrap.solveInPlace(sol_wrap);

//   BOOST_CHECK(sol_wrap.isApprox(sol_eig));
//   BOOST_CHECK(ldlt_wrap.matrixLDLT().isApprox(ldlt.matrixLDLT()));
// }

// BOOST_FIXTURE_TEST_CASE(test_dense_ldlt_ours, ldlt_test_fixture,
//                         *utf::tolerance(TOL)) {
//   // dense LDLT
//   DenseLDLT<Scalar> dense_ldlt(mat);
//   BOOST_REQUIRE(dense_ldlt.info() == Eigen::Success);

//   MatrixXs reconstr = dense_ldlt.reconstructedMatrix();
//   BOOST_CHECK(reconstr.isApprox(mat));

//   MatrixXs sol_dense = rhs;
//   dense_ldlt.solveInPlace(sol_dense);

//   Scalar dense_err = math::infty_norm(sol_dense - sol_eig);
//   fmt::print("Dense err = {:.5e}\n", dense_err);
//   BOOST_CHECK(sol_dense.isApprox(sol_eig, TOL));
//   BOOST_CHECK(rhs.isApprox(mat * sol_dense));
// }

// BOOST_FIXTURE_TEST_CASE(test_bunchkaufman, ldlt_test_fixture,
//                         *utf::tolerance(TOL)) {
//   // dense LDLT
//   Eigen::BunchKaufman<MatrixXs, Eigen::Lower> lblt(mat);

//   // MatrixXs reconstr = dense_ldlt.reconstructedMatrix();
//   // BOOST_CHECK(reconstr.isApprox(mat));

//   MatrixXs sol_dense = rhs;
//   lblt.solveInPlace(sol_dense);

//   Scalar dense_err = math::infty_norm(sol_dense - sol_eig);
//   fmt::print("BunchKaufman err = {:.5e}\n", dense_err);
//   BOOST_CHECK(sol_dense.isApprox(sol_eig, TOL));
//   BOOST_CHECK(rhs.isApprox(mat * sol_dense));
// }

// BOOST_FIXTURE_TEST_CASE(test_block_ldlt_ours, ldlt_test_fixture,
//                         *utf::tolerance(TOL)) {
//   fmt::print("Input matrix pattern:\n");
//   linalg::print_sparsity_pattern(sym_mat);
//   BOOST_REQUIRE(sym_mat.check_if_symmetric());

//   BlockLDLT<Scalar> block_permuted(size, sym_mat);
//   block_permuted.findSparsifyingPermutation();
//   block_permuted.compute(mat);
//   auto best_perm = block_permuted.blockPermIndices();
//   fmt::print("Optimal permutation: {}\n",
//              fmt::join(best_perm.begin(), best_perm.end(), ", "));

//   {
//     auto copy_sym = sym_mat.copy();
//     linalg::symbolic_deep_copy(sym_mat, copy_sym, best_perm.data());
//     fmt::print("Permuted structure:\n");
//     linalg::print_sparsity_pattern(copy_sym);
//   }

//   fmt::print("Optimized structure (nnz={:d}):\n",
//              block_permuted.structure().count_nnz());
//   linalg::print_sparsity_pattern(block_permuted.structure());

//   auto pmat = block_permuted.permutationP();
//   fmt::print("Permutation matrix: {}\n", pmat.indices().transpose());

//   Eigen::ComputationInfo info = block_permuted.info();
//   BOOST_REQUIRE(info == Eigen::Success);

//   {
//     auto copy_sym_mat = sym_mat.copy();
//     copy_sym_mat.llt_in_place();
//     fmt::print("Un-permuted (suboptimal) LLT (nnz={:d}):\n",
//                copy_sym_mat.count_nnz());
//     linalg::print_sparsity_pattern(copy_sym_mat);
//   }

//   MatrixXs reconstr = block_permuted.reconstructedMatrix();
//   BOOST_CHECK(reconstr.isApprox(mat));
//   auto reconstr_err = math::infty_norm(reconstr - mat);
//   fmt::print("Block reconstr err = {:.5e}\n", reconstr_err);

//   MatrixXs sol_block = rhs;
//   block_permuted.solveInPlace(sol_block);
//   Scalar err = math::infty_norm(sol_block - sol_eig);
//   fmt::print("err = {:.5e}\n", err);
//   BOOST_CHECK(sol_block.isApprox(sol_eig, TOL));
//   BOOST_CHECK(rhs.isApprox(mat * sol_block));
// }

// BOOST_AUTO_TEST_CASE(block_structure_allocator) {

//   std::vector<isize> nprims = {7, 14};
//   std::vector<isize> nduals = {14};

//   BOOST_TEST_MESSAGE("Default structure:");
//   linalg::SymbolicBlockMatrix default_structure =
//       proxnlp::create_default_block_structure(nprims, nduals);
//   linalg::print_sparsity_pattern(default_structure);

//   BOOST_TEST_MESSAGE("Modified structure:");
//   linalg::SymbolicBlockMatrix modified_structure = default_structure.copy();

//   // zero out the primal off-diagonal blocks
//   for (isize i = 0; i < (unsigned int)nprims.size(); ++i) {
//     for (isize j = 0; i < (unsigned int)nprims.size(); ++i) {
//       if (i != j) {
//         modified_structure(i, j) = linalg::Zero;
//         modified_structure(j, i) = linalg::Zero;
//       }
//     }
//   }

//   linalg::print_sparsity_pattern(modified_structure);
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
// BOOST_FIXTURE_TEST_CASE(test_proxsuite_ldlt, ldlt_test_fixture,
//                         *utf::tolerance(TOL)) {
//   linalg::ProxSuiteLDLTWrapper<Scalar> ps_ldlt(mat.rows(), rhs.cols() + 1);
//   ps_ldlt.compute(mat);
//   MatrixXs reconstr = ps_ldlt.reconstructedMatrix();
//   BOOST_CHECK(reconstr.isApprox(mat));

//   auto sol_ps = rhs;
//   ps_ldlt.solveInPlace(sol_ps);

//   BOOST_CHECK(sol_ps.isApprox(sol_eig));
//   Scalar solve_err = math::infty_norm(sol_ps - sol_eig);
//   fmt::print("proxsuite_err = {:.5e}\n", solve_err);
// }

// #endif

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
