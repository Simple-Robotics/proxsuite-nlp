#include "lienlp/constraint-base.hpp"
#include "lienlp/modelling/constraints/equality-constraint.hpp"
#include "lienlp/modelling/constraints/negative-orthant.hpp"
// #include "lienlp/modelling/constraints/l1-penalty.hpp"

#include "lienlp/modelling/spaces/pinocchio-groups.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(constraint)

using namespace lienlp;

const int N = 20;
using Vs_ = pinocchio::VectorSpaceOperationTpl<N, double>;
PinocchioLieGroup<Vs_> space;

BOOST_AUTO_TEST_CASE(test_equality)
{
  auto x0 = space.zero();
  auto x1 = space.rand();
}


BOOST_AUTO_TEST_SUITE_END()
