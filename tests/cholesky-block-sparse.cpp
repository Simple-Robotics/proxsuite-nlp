/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, ",", "\n", "[", "]")

#include "util.hpp"
#include "proxsuite-nlp/ldlt-allocator.hpp"

#include <boost/test/unit_test.hpp>

#include "proxsuite-nlp/math.hpp"
#include "proxsuite-nlp/fmt-eigen.hpp"

BOOST_AUTO_TEST_SUITE(cholesky_sparse)

namespace utf = boost::unit_test;
using namespace proxsuite::nlp;

using linalg::BlockLDLT;
using linalg::DenseLDLT;

constexpr isize n = 3;
constexpr double TOL = 1e-11;
constexpr double TOL_LOOSE = 1e-10;
const isize ndx = 24;

auto create_problem_structure() -> linalg::SymbolicBlockMatrix {
  // clang-format off
  BlockKind *data = new BlockKind[n * n]{
      BlockKind::Diag,  BlockKind::Dense, BlockKind::Dense,
      BlockKind::Dense, BlockKind::Dense, BlockKind::Diag,
      BlockKind::Dense, BlockKind::Diag, BlockKind::Diag
  };
  // clang-format on

  // isize row_segments[n] = {8, 16, 16};
  isize *row_segments = new isize[n]{12, ndx, ndx};
  return {data, row_segments, n, n};
}

auto sym_mat = create_problem_structure();
isize ncols = ndx;

struct ldlt_test_fixture {
  ldlt_test_fixture() : mat(), rhs(), ldlt() { this->init(); }
  ~ldlt_test_fixture() = default;

  MatrixXs mat;
  MatrixXs rhs;
  Eigen::LDLT<MatrixXs> ldlt;
  MatrixXs sol_eig;
  isize size;
  Eigen::VectorXi signature;

  void init() {
    mat = getRandomSymmetricBlockMatrix(sym_mat);
    ldlt.compute(mat);
    size = mat.cols();
    rhs = MatrixXs::Random(size, ncols);
    sol_eig = ldlt.solve(rhs);
    ComputeSignatureVisitor{signature}(ldlt);
  }
};

BOOST_FIXTURE_TEST_CASE(test_eigen_ldlt, ldlt_test_fixture,
                        *utf::tolerance(TOL)) {
  BOOST_REQUIRE(ldlt.info() == Eigen::Success);

  MatrixXs reconstr = ldlt.reconstructedMatrix();
  BOOST_CHECK(reconstr.isApprox(mat));

  MatrixXs sol_wrap = rhs;
  ldlt.solveInPlace(sol_wrap);

  BOOST_CHECK(sol_wrap.isApprox(sol_eig));
  BOOST_CHECK(ldlt.matrixLDLT().isApprox(ldlt.matrixLDLT()));
  auto t = computeInertiaTuple(signature);
  fmt::print("Signature: {}\n", fmt::join(t, " "));
}

BOOST_FIXTURE_TEST_CASE(test_dense_ldlt_ours, ldlt_test_fixture,
                        *utf::tolerance(TOL_LOOSE)) {
  // dense LDLT
  DenseLDLT<Scalar> dense_ldlt(mat);
  BOOST_REQUIRE(dense_ldlt.info() == Eigen::Success);

  MatrixXs reconstr = dense_ldlt.reconstructedMatrix();
  BOOST_CHECK(reconstr.isApprox(mat));

  MatrixXs sol_dense = rhs;
  dense_ldlt.solveInPlace(sol_dense);

  Scalar dense_err = math::infty_norm(sol_dense - sol_eig);
  fmt::print("Dense err = {:.5e}\n", dense_err);
  BOOST_CHECK(sol_dense.isApprox(sol_eig, TOL_LOOSE));
  BOOST_CHECK(rhs.isApprox(mat * sol_dense, TOL_LOOSE));
}

BOOST_FIXTURE_TEST_CASE(test_bunchkaufman, ldlt_test_fixture,
                        *utf::tolerance(TOL)) {
  Eigen::BunchKaufman<MatrixXs, Eigen::Lower> lblt(mat);

  MatrixXs sol_dense = rhs;
  lblt.solveInPlace(sol_dense);

  Scalar dense_err = math::infty_norm(sol_dense - sol_eig);
  fmt::print("BunchKaufman err = {:.5e}\n", dense_err);
  BOOST_CHECK(sol_dense.isApprox(sol_eig));
  BOOST_CHECK(rhs.isApprox(mat * sol_dense));

  auto sg = signature; // copy
  internal::bunch_kaufman_compute_signature(lblt, sg);
  std::array<int, 3> t_eigen = computeInertiaTuple(signature);
  std::array<int, 3> t_bk = computeInertiaTuple(sg);
  fmt::print("Eig. Signature: {}\n", fmt::join(t_eigen, " "));
  fmt::print("BK Signature: {}\n", fmt::join(t_bk, " "));

  BOOST_CHECK((t_eigen[0] == t_bk[0]) && (t_eigen[1] == t_bk[1]) &&
              (t_eigen[2] == t_bk[2]));
}

BOOST_FIXTURE_TEST_CASE(test_block_ldlt_ours, ldlt_test_fixture,
                        *utf::tolerance(TOL)) {
  fmt::print("Input matrix pattern:\n");
  linalg::print_sparsity_pattern(sym_mat);
  BOOST_REQUIRE(sym_mat.check_if_symmetric());

  BlockLDLT<Scalar> block_permuted(size, sym_mat);
  block_permuted.findSparsifyingPermutation();
  block_permuted.compute(mat);
  auto best_perm = block_permuted.blockPermIndices();
  fmt::print("Optimal permutation: {}\n",
             fmt::join(best_perm.begin(), best_perm.end(), ", "));

  {
    auto copy_sym = sym_mat.copy();
    linalg::symbolic_deep_copy(sym_mat, copy_sym, best_perm.data());
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

BOOST_AUTO_TEST_CASE(block_structure_allocator) {

  std::vector<isize> nprims = {7, 14};
  std::vector<isize> nduals = {14};

  BOOST_TEST_MESSAGE("Default structure:");
  linalg::SymbolicBlockMatrix default_structure =
      proxsuite::nlp::create_default_block_structure(nprims, nduals);
  linalg::print_sparsity_pattern(default_structure);

  BOOST_TEST_MESSAGE("Modified structure:");
  linalg::SymbolicBlockMatrix modified_structure = default_structure.copy();

  // zero out the primal off-diagonal blocks
  for (isize i = 0; i < (unsigned int)nprims.size(); ++i) {
    for (isize j = 0; i < (unsigned int)nprims.size(); ++i) {
      if (i != j) {
        modified_structure(i, j) = linalg::Zero;
        modified_structure(j, i) = linalg::Zero;
      }
    }
  }

  linalg::print_sparsity_pattern(modified_structure);
}

#ifdef PROXSUITE_NLP_USE_PROXSUITE_LDLT

BOOST_FIXTURE_TEST_CASE(test_proxsuite_ldlt, ldlt_test_fixture,
                        *utf::tolerance(TOL)) {
  linalg::ProxSuiteLDLTWrapper<Scalar> ps_ldlt(mat.rows(), rhs.cols() + 1);
  ps_ldlt.compute(mat);
  MatrixXs reconstr = ps_ldlt.reconstructedMatrix();
  BOOST_CHECK(reconstr.isApprox(mat));

  auto sol_ps = rhs;
  ps_ldlt.solveInPlace(sol_ps);

  BOOST_CHECK(sol_ps.isApprox(sol_eig));
  Scalar solve_err = math::infty_norm(sol_ps - sol_eig);
  fmt::print("proxsuite_err = {:.5e}\n", solve_err);
}

#endif

int main(int argc, char **argv) {
  // call default test initialization function
  // see Boost.Test docs:
  // https://www.boost.org/doc/libs/1_80_0/libs/test/doc/html/boost_test/adv_scenarios/shared_lib_customizations/entry_point.html
  int tests_result = utf::unit_test_main(&init_unit_test, argc, argv);

  return tests_result;
}

BOOST_AUTO_TEST_SUITE_END()
