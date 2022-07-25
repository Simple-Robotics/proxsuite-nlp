#include "proxnlp/modelling/costs/squared-distance.hpp"
#include "proxnlp/cost-sum.hpp"
#include "proxnlp/modelling/spaces/pinocchio-groups.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(cost)

#ifdef WITH_PINOCCHIO
using namespace proxnlp;
namespace pin = pinocchio;
namespace utf = boost::unit_test;
using SE2 = PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<2, double>>;

BOOST_AUTO_TEST_CASE(test_cost_sum, *utf::tolerance(1e-10)) {
  SE2 space;
  auto x0 = space.neutral();
  auto x1 = space.rand();
  auto x2 = space.rand();

  QuadraticDistanceCost<double> cost1(space, x0);
  QuadraticDistanceCost<double> cost2(space, x1);
  CostSum<double> cost_sum = cost1 + cost2;

  BOOST_CHECK_EQUAL(cost_sum.call(x2), cost1.call(x2) + cost2.call(x2));

  cost_sum += cost1; // operator+=(lvalue costfunctionbase&)
  BOOST_TEST(cost_sum.call(x2) == 2 * cost1.call(x2) + cost2.call(x2));

  CostSum<double> c3 =
      cost1 + cost2 + cost2; // invokes operator+ with rvalue ref
  BOOST_TEST(c3.call(x2) == cost1.call(x2) + 2 * cost2.call(x2));

  fmt::print("c3(x2): {:.3f}\n", c3.call(x2));
  c3 *= .5;
  fmt::print("c3(x2): {:.3f}\n", c3.call(x2));
  auto c4 =
      .5 * cost1 + cost2; // invokes operator+ with CostSum&& lhs, should be c3
  fmt::print("c4(x2): {:.3f}\n", c4.call(x2));
  BOOST_TEST(c3.call(x2) == c4.call(x2));
}
#endif

BOOST_AUTO_TEST_SUITE_END()
