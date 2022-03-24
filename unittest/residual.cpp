#include "lienlp/modelling/residuals/linear.hpp"

#include <Eigen/Core>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>


BOOST_AUTO_TEST_SUITE(residual)

using namespace lienlp;

BOOST_AUTO_TEST_CASE(test_linear)
{
  Eigen::Matrix4d A;
  A.setIdentity();
  A.topLeftCorner(2, 2).setConstant(0.5);
  Eigen::Vector4d b;
  b.setRandom();
  LinearResidual<double> res(A, b);

  Eigen::Vector4d x0, x1;
  x0.setZero();
  x1.setOnes();
  fmt::print("res(x0)  = {}\n", res(x0));
  fmt::print("res(x1)  = {}\n", res(x1));

  Eigen::Matrix4d J0;
  res.computeJacobian(x0, J0);
  fmt::print("{}  <<  Jres(x0)\n", J0);

  BOOST_CHECK(res(x0).isApprox(b));
  BOOST_CHECK(J0.isApprox(A));

}

BOOST_AUTO_TEST_SUITE_END()
