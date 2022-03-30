#include "lienlp/modelling/costs/squared-distance.hpp"
#include "lienlp/cost-sum.hpp"
#include "lienlp/modelling/spaces/pinocchio-groups.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(cost)

using namespace lienlp;
namespace pin = pinocchio;
using SE2 = PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<2, double>>;


BOOST_AUTO_TEST_CASE(test_cost_sum)
{
  SE2 space;
  auto x0 = space.neutral();
  auto x1 = space.rand();
  auto x2 = space.rand();

  QuadDistanceCost<double> cost1(space, x0);
  QuadDistanceCost<double> cost2(space, x1);
  auto cost_sum = cost1 + cost2;

  BOOST_CHECK_EQUAL(cost_sum.call(x2), cost1.call(x2) + cost2.call(x2));

  cost_sum += cost1;
  BOOST_CHECK_EQUAL(cost_sum.call(x2), 2 * cost1.call(x2) + cost2.call(x2));
}


BOOST_AUTO_TEST_SUITE_END()
