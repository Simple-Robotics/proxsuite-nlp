/// @file
/// @author Wilson Jallet
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, ",", "\n", "[", "]")

#include "block-test.hpp"
#include <random>

#include <benchmark/benchmark.h>
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include <boost/mpl/vector.hpp>

namespace utf = boost::unit_test;

using linalg::TriangularBlockMatrix;

// const int RSEED = 42;

struct prob_data {
  static constexpr isize n = 3;

  // clang-format off
  BlockKind data[n * n] = {
      BlockKind::TriL,  BlockKind::Dense, BlockKind::Zero,
      BlockKind::Zero, BlockKind::Diag, BlockKind::TriL,
      BlockKind::Zero, BlockKind::Zero, BlockKind::Diag
  };
  // clang-format on

  isize row_segments[n];
  isize size;

  SymbolicBlockMatrix sym_mat;
  MatrixXs rhs;
  MatrixXs mat;

  prob_data() : sym_mat{data, row_segments, n, n} {

    // std::random_device rd;
    // std::mt19937 randeng(rd());
    // randeng.seed(RSEED);
    // std::uniform_int_distribution<isize> randdist(2, 20);

    // auto gen = [&]() -> isize { return randdist(randeng); };

    // std::generate(row_segments, row_segments + n, gen);
    row_segments[0] = 20;
    row_segments[1] = 20;
    row_segments[2] = 20;

    size = std::accumulate(row_segments, row_segments + n, isize(0));
    isize ncols_rhs = size;
    rhs = MatrixXs::Random(size, ncols_rhs);
    mat = get_block_matrix(sym_mat);
  }
};

constexpr isize prob_data::n;

template <int _Mode> struct tri_fixture : prob_data {
  enum { Mode = _Mode };
  tri_fixture() : prob_data() {
    constexpr bool IsUpper = (Mode & Eigen::Upper) == Eigen::Upper;
    if (IsUpper)
      sym_mat = sym_mat.transpose();
    mat = get_block_matrix(sym_mat);
    sol_eig = view().solve(rhs);
  }

  MatrixXs sol_eig;
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
  linalg::print_sparsity_pattern(Fix::sym_mat);

  TriangularBlockMatrix<MatrixXs, Fix::Mode> tri_mat(loc_mat, Fix::sym_mat);

  MatrixXs sol_ours = Fix::rhs;
  bool flag = tri_mat.solveInPlace(sol_ours);

  BOOST_REQUIRE(flag);

  BOOST_CHECK(Fix::sol_eig.isApprox(sol_ours));
}

template <int Mode> void BM_block_tri_solve(benchmark::State &state) {
  tri_fixture<Mode> fix{};
  MatrixXs loc_mat = fix.view();
  TriangularBlockMatrix<MatrixXs, Mode> tri_block_mat(loc_mat, fix.sym_mat);

  for (auto _ : state) {
    auto sol_ours = fix.rhs;
    bool flag = tri_block_mat.solveInPlace(sol_ours);
    if (!flag) {
      state.SkipWithError("Solve in place failed.");
      break;
    }
    if (!sol_ours.isApprox(fix.sol_eig)) {
      state.SkipWithError("Got wrong solution.");
      break;
    }
  }
}

template <int Mode> void BM_tri_solve(benchmark::State &state) {
  tri_fixture<Mode> fix{};
  auto loc_mat_tri_view = fix.view();

  for (auto _ : state) {
    auto sol_ours = fix.rhs;
    loc_mat_tri_view.solveInPlace(sol_ours);
    if (!sol_ours.isApprox(fix.sol_eig)) {
      state.SkipWithError("Got wrong solution.");
      break;
    }
  }
}

const auto unit = benchmark::kMicrosecond;
BENCHMARK_TEMPLATE(BM_block_tri_solve, Eigen::Lower)->Unit(unit);
BENCHMARK_TEMPLATE(BM_block_tri_solve, Eigen::UnitLower)->Unit(unit);
BENCHMARK_TEMPLATE(BM_block_tri_solve, Eigen::Upper)->Unit(unit);
BENCHMARK_TEMPLATE(BM_block_tri_solve, Eigen::UnitUpper)->Unit(unit);
BENCHMARK_TEMPLATE(BM_tri_solve, Eigen::Lower)->Unit(unit);
BENCHMARK_TEMPLATE(BM_tri_solve, Eigen::UnitLower)->Unit(unit);
BENCHMARK_TEMPLATE(BM_tri_solve, Eigen::Upper)->Unit(unit);
BENCHMARK_TEMPLATE(BM_tri_solve, Eigen::UnitUpper)->Unit(unit);

int main(int argc, char **argv) {

  // call default test initialization function
  // see Boost.Test docs:
  // https://www.boost.org/doc/libs/1_80_0/libs/test/doc/html/boost_test/adv_scenarios/shared_lib_customizations/entry_point.html
  bool tests_result = utf::unit_test_main(&init_unit_test, argc, argv);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return tests_result;
}
