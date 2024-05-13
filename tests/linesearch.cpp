#include <boost/test/unit_test.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/polynomial.hpp>

#include "proxsuite-nlp/linesearch-armijo.hpp"

using namespace proxsuite::nlp;
using boost::math::tools::polynomial;

BOOST_AUTO_TEST_CASE(armijo) {
  polynomial<double> p1{{1, 1}};
  polynomial<double> p2{{-0.3, 1}};
  polynomial<double> p = p1 * p2 * p2;
  std::cout << p << std::endl;
  polynomial<double> dp = p.prime();
  std::cout << dp << std::endl;

  Linesearch<double>::Options opts;
  opts.contraction_min = 0.1;
  ArmijoLinesearch<double> ls{opts};
  double alpha = 0.;
  double dphi0 = dp(0.);
  ls.run(p, p(0.), dphi0, alpha);

  fmt::print("Found alpha = {:.4e}\n", alpha);

  auto [r0, r1] = boost::math::tools::quadratic_roots(
      dp.data()[2], dp.data()[1], dp.data()[0]);
  fmt::print("Derivative roots: {:.4e} / {:.4e}\n", r0, r1);
}
