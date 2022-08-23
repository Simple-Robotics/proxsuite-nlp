#include "proxnlp/math.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(math)

BOOST_AUTO_TEST_CASE(infty_norm) {
  Eigen::Vector3d x0;
  x0.setRandom();

  double xnorm = proxnlp::math::infty_norm(x0);
  BOOST_CHECK_EQUAL(xnorm, x0.lpNorm<Eigen::Infinity>());

  std::vector<Eigen::Vector3d> xv{10};
  for (std::size_t i = 0; i < xv.size(); i++) {
    xv[i].setRandom();
  }
  double vnorm = proxnlp::math::infty_norm(xv);
  double t = 0.;
  for (std::size_t i = 0; i < xv.size(); i++) {
    t = std::max(t, xv[i].lpNorm<Eigen::Infinity>());
  }
  BOOST_CHECK_EQUAL(vnorm, t);
}

BOOST_AUTO_TEST_SUITE_END()
