#include "proxnlp/modelling/costs/squared-distance.hpp"
#include "proxnlp/cost-sum.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <boost/test/unit_test.hpp>

#ifdef PROXNLP_WITH_PINOCCHIO
#include "proxnlp/modelling/spaces/pinocchio-groups.hpp"
#endif

using namespace proxnlp;
namespace utf = boost::unit_test;
#ifdef PROXNLP_WITH_PINOCCHIO
namespace pin = pinocchio;
using SE2 = PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<2, double>>;
#endif

using Scalar = double;
using CostBase = CostFunctionBaseTpl<Scalar>;
using CostPtr = shared_ptr<CostBase>;
using C2Function = C2FunctionTpl<Scalar>;

struct CustomC2Func : C2FunctionTpl<Scalar> {

  CustomC2Func() : C2FunctionTpl<Scalar>(SE2(), 1) {}
  VectorXs operator()(const ConstVectorRef &) const {
    return VectorXs::Random(nr());
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const {
    Jout.setZero();
  }
};

auto function_that_takes_a_cost(const CostPtr &) {
  // does nothing
}

BOOST_AUTO_TEST_SUITE(cost)

#ifdef PROXNLP_WITH_PINOCCHIO

BOOST_AUTO_TEST_CASE(test_cost_sum, *utf::tolerance(1e-10)) {
  auto space = std::make_shared<SE2>();
  auto x0 = space->neutral();
  auto x1 = space->rand();
  auto x2 = space->rand();

  CostPtr cost1 = std::make_shared<QuadraticDistanceCostTpl<double>>(space, x0);
  CostPtr cost2 = std::make_shared<QuadraticDistanceCostTpl<double>>(space, x1);
  auto cost_sum = cost1 + cost2;

  BOOST_CHECK_EQUAL(cost_sum->call(x2), cost1->call(x2) + cost2->call(x2));

  *cost_sum += cost1; // operator+=(lvalue costfunctionbase&)
  BOOST_TEST(cost_sum->call(x2) == 2 * cost1->call(x2) + cost2->call(x2));

  auto c3 = cost1 + cost2 + cost2; // invokes operator+
  BOOST_TEST(c3->call(x2) == cost1->call(x2) + 2 * cost2->call(x2));

  fmt::print("c3(x2): {:.3f}\n", c3->call(x2));
  *c3 *= .5;
  fmt::print("c3(x2): {:.3f}\n", c3->call(x2));
  auto c4 =
      .5 * cost1 + cost2; // invokes operator+ with CostSum&& lhs, should be c3
  fmt::print("c4(x2): {:.3f}\n", c4->call(x2));
  BOOST_TEST(c3->call(x2) == c4->call(x2));
}

BOOST_AUTO_TEST_CASE(test_cast_cost) {
  auto space = std::make_shared<SE2>();
  auto fun = std::make_shared<CustomC2Func>();
  CostPtr cost = downcast_function_to_cost<Scalar>(fun);

  BOOST_TEST(cost != nullptr);

  function_that_takes_a_cost(cost);
}

#endif

BOOST_AUTO_TEST_SUITE_END()
