#include "lienlp/modelling/residuals/linear.hpp"
#include "lienlp/function-ops.hpp"
#include "lienlp/modelling/costs/squared-distance.hpp"
#include "lienlp/modelling/spaces/pinocchio-groups.hpp"

#include <Eigen/Core>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>


BOOST_AUTO_TEST_SUITE(residual)

using namespace lienlp;

BOOST_AUTO_TEST_CASE(test_linear)
{
  BOOST_TEST_MESSAGE("Test linear function:");
  Eigen::Matrix4d A;
  A.setIdentity();
  A.topLeftCorner(2, 2).setConstant(0.5);
  Eigen::Vector4d b;
  b.setRandom();
  LinearFunction<double> res(A, b);

  Eigen::Vector4d x0, x1;
  x0.setZero();
  x1.setOnes();
  fmt::print("res(x0)  = {}\n", res(x0).transpose());
  fmt::print("res(x1)  = {}\n", res(x1).transpose());

  Eigen::Matrix4d J0;
  res.computeJacobian(x0, J0);
  fmt::print("{}  <<  Jres(x0)\n", J0);

  BOOST_CHECK(res(x0).isApprox(b));
  BOOST_CHECK(J0.isApprox(A));

}

BOOST_AUTO_TEST_CASE(test_compose)
{
  BOOST_TEST_MESSAGE("Test function composition:");
  const int N = 3;
  using Matrix_t = Eigen::Matrix<double, N, N>;
  using Vector_t = Eigen::Matrix<double, N, 1>;
  
  Matrix_t A;
  Vector_t b;
  A.setRandom();
  b.setRandom();
  using ResType = LinearFunction<double>;
  ResType res1(A, b);
  
  Vector_t x0;
  x0.setOnes();

  ComposeFunctionTpl<double> res2(res1, res1);

  fmt::print("A matrix:\n{}\n", A);
  fmt::print("A squared:\n{}\n", A * A);
  Eigen::MatrixXd J_(N, N);
  res2.computeJacobian(x0, J_);
  fmt::print("compose Jx0:\n{}\n", J_);
  BOOST_CHECK(J_.isApprox(A * A));

  auto v0 = res2(x0);
  auto v1_manual = res1(res1(x0));
  fmt::print("f(f(x0)):  {}\nshould be: {}\n",
             v0.transpose(), v1_manual.transpose());
  BOOST_CHECK(v1_manual.isApprox(v0));

}

BOOST_AUTO_TEST_SUITE_END()
