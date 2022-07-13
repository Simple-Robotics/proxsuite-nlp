#include "proxnlp/constraint-base.hpp"
#include "proxnlp/modelling/constraints/equality-constraint.hpp"
#include "proxnlp/modelling/constraints/negative-orthant.hpp"
// #include "proxnlp/modelling/constraints/l1-penalty.hpp"

#include "proxnlp/modelling/spaces/vector-space.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(constraint)

using namespace proxnlp;

const int N = 20;
VectorSpaceTpl<double> space(N);
PROXNLP_DYNAMIC_TYPEDEFS(double);

BOOST_AUTO_TEST_CASE(test_equality) {
  VectorXs x0 = space.neutral();
  VectorXs x1 = space.rand();
  VectorXs zout(N);
  zout.setZero();

  EqualityConstraint<double> eq_set;
  double mu = 0.1;

  double m = computeMoreauEnvelope(eq_set, x1, 1. / mu, zout);
  BOOST_TEST_CHECK(zout.isApprox(x1));
  BOOST_TEST_CHECK(m == (0.5 / mu * zout.squaredNorm()));
}

BOOST_AUTO_TEST_SUITE_END()
