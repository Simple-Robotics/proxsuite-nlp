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
  CostSum<double> cost_sum = cost1 + cost2;

  BOOST_CHECK_EQUAL(cost_sum.call(x2), cost1.call(x2) + cost2.call(x2));

  cost_sum += cost1; // operator+=(lvalue costfunctionbase&)
  BOOST_CHECK_EQUAL(cost_sum.call(x2), 2 * cost1.call(x2) + cost2.call(x2));

  CostSum<double> c3 = cost1 + cost2 + cost2; // invokes operator+ with rvalue ref
  BOOST_CHECK_EQUAL(c3.call(x2), cost1.call(x2) + 2 * cost2.call(x2));

  fmt::print("c3(x2): {:.3f}\n", c3.call(x2));
  c3 *= .5;
  fmt::print("c3(x2): {:.3f}\n", c3.call(x2));
  auto c4 = .5 * cost1 + cost2;  // invokes operator+ with CostSum&& lhs, should be c3
  fmt::print("c4(x2): {:.3f}\n", c4.call(x2));
}


BOOST_AUTO_TEST_SUITE_END()
