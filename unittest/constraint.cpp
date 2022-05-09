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

BOOST_AUTO_TEST_CASE(test_equality)
{
  VectorXs x0 = space.neutral();
  VectorXs x1 = space.rand();


}


BOOST_AUTO_TEST_SUITE_END()
