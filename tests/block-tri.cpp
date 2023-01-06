/// @file
/// @author Wilson Jallet
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, ",", "\n", "[", "]")

#include "block-test.hpp"
#include <random>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include <boost/mpl/vector.hpp>

namespace utf = boost::unit_test;

using linalg::TriangularBlockMatrix;

struct prob_data {
  static constexpr isize n = 3;

  // clang-format off
  BlockKind data[n * n] = {
      BlockKind::TriL,  BlockKind::Dense, BlockKind::TriL,
      BlockKind::Zero, BlockKind::Diag, BlockKind::TriL,
      BlockKind::Zero, BlockKind::Zero, BlockKind::Diag
  };
  // clang-format on

  isize row_segments[n];
  isize size;

  SymbolicBlockMatrix sym_mat;
  VectorXs rhs;

  prob_data() : sym_mat{data, row_segments, n, n} {

    std::random_device rd;
    std::mt19937 randeng(rd());
    randeng.seed(42);
    std::uniform_int_distribution<isize> randdist(1, 20);

    auto gen = [&]() -> isize { return randdist(randeng); };

    std::generate(row_segments, row_segments + n, gen);
    fmt::print("row_segs = {}\n", fmt::join(row_segments, ", "));

    size = std::accumulate(row_segments, row_segments + n, isize(0));
    rhs = VectorXs::Random(size);
  }
};

constexpr isize prob_data::n;

template <int _Mode> struct tri_fixture : prob_data {
  enum { Mode = _Mode };
  tri_fixture() : prob_data() {
    constexpr bool IsUpper = (Mode & Eigen::Upper) == Eigen::Upper;
    if (IsUpper)
      sym_mat = sym_mat.transpose();
    linalg::print_sparsity_pattern(sym_mat);
    mat = get_block_matrix(sym_mat);
    sol_eig = view().solve(rhs);
  }

  MatrixXs mat;
  VectorXs sol_eig;
  auto view() { return mat.triangularView<Mode>(); }
};

// clang-format off
using test_modes = boost::mpl::vector<
      tri_fixture<Eigen::Lower>,
      tri_fixture<Eigen::UnitLower>,
      tri_fixture<Eigen::Upper>,
      tri_fixture<Eigen::UnitUpper>>;
// clang-format on

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_block_tri, Fix, test_modes, Fix) {

  MatrixXs loc_mat = Fix::view();

  TriangularBlockMatrix<MatrixXs, Fix::Mode> tri_mat(loc_mat, Fix::sym_mat);

  VectorXs sol_ours = Fix::rhs;
  bool flag = tri_mat.solveInPlace(sol_ours);

  BOOST_REQUIRE(flag);

  BOOST_CHECK(Fix::sol_eig.isApprox(sol_ours));
  fmt::print("Sol eigen: {}\n", Fix::sol_eig.transpose());
  fmt::print("Sol ours : {}\n", sol_ours.transpose());
}

int main(int argc, char **argv) {

  // call default test initialization function
  // see Boost.Test docs:
  // https://www.boost.org/doc/libs/1_80_0/libs/test/doc/html/boost_test/adv_scenarios/shared_lib_customizations/entry_point.html
  return utf::unit_test_main(&init_unit_test, argc, argv);
}
