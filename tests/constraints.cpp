#include "proxsuite-nlp/constraint-base.hpp"
#include "proxsuite-nlp/modelling/constraints/equality-constraint.hpp"
#include "proxsuite-nlp/modelling/constraints/negative-orthant.hpp"
// #include "proxsuite-nlp/modelling/constraints/l1-penalty.hpp"

#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(constraint)

using namespace proxsuite::nlp;

const int N = 20;
VectorSpaceTpl<double> space(N);
PROXSUITE_NLP_DYNAMIC_TYPEDEFS(double);

BOOST_AUTO_TEST_CASE(test_equality) {
  VectorXs x0 = space.neutral();
  VectorXs x1 = space.rand();
  VectorXs zout(N);
  zout.setZero();

  EqualityConstraintTpl<double> eq_set;
  double mu = 0.1;

  eq_set.setProxParameter(mu);
  double m = eq_set.computeMoreauEnvelope(x1, zout);
  BOOST_TEST_CHECK(zout.isApprox(x1));
  BOOST_TEST_CHECK(m == (0.5 / mu * zout.squaredNorm()));
}

BOOST_AUTO_TEST_SUITE_END()
