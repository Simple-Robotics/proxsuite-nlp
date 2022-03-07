#include "lienlp/manifold-base.hpp"
#include "lienlp/pinocchio-groups.hpp"

// #ifdef WITH_PINOCCHIO_SUPPORT
  #include <pinocchio/parsers/sample-models.hpp>
  #include <pinocchio/multibody/liegroup/vector-space.hpp>
// #endif

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>


BOOST_AUTO_TEST_SUITE(manifold)

BOOST_AUTO_TEST_CASE(test_lg_vecspace)
{
  std::cout << "test RN" << '\n';
  const int N = 4;
  using Vs = pinocchio::VectorSpaceOperationTpl<N, double>;
  const Vs lg;
  lienlp::PinocchioLieGroup<Vs> space(lg);
  Vs::ConfigVector_t x0(space.get_nq());
  Vs::TangentVector_t v(space.get_nv());

  auto x1 = space.integrate(x0, v);
  std::cout << x1 << std::endl;

}


// #ifdef WITH_PINOCCHIO_SUPPORT
BOOST_AUTO_TEST_CASE(test_pinmodel)
{
  std::cout << "test_pinmodel" << '\n';

  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model, true);

  using M = lienlp::MultibodyConfiguration<double>;
  M space(model);
  typedef typename M::Point_t Vec_t;
  Vec_t x0 = pinocchio::neutral(model);
  Vec_t v(model.nv);
  v.setRandom();
  std::cout << v << std::endl;

  Vec_t xout(model.nq);
  space.integrate(x0, v, xout);
  std::cout << "xout (1):\n" << xout << std::endl;

  Vec_t x1 = pinocchio::randomConfiguration(model);
  Vec_t d(model.nv);
  space.diff(x0, x1, d);
  std::cout << "xout (2):\n" << d << std::endl;
}
// #endif

BOOST_AUTO_TEST_SUITE_END()
