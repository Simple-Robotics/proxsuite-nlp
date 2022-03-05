#include "lienlp/manifold-base.hpp"

#include "lienlp/pinocchio-groups.hpp"

// #ifdef WITH_PINOCCHIO_SUPPORT
  #include <pinocchio/parsers/sample-models.hpp>
// #endif

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>


BOOST_AUTO_TEST_SUITE(manifold)

BOOST_AUTO_TEST_CASE(test_vector)
{
  const int nq = 2;
  typedef lienlp::VectorSpace<nq, double> M;
  M space;
  M::Vec_t x0;
  M::Vec_t x1;
  x0.setZero();
  x1.setRandom();

  std::cout << x0 << std::endl;

  M::Vec_t out;

  space.diff(x0, x1, out);
  std::cout << "what:" << out << std::endl;

}

// #ifdef WITH_PINOCCHIO_SUPPORT
  BOOST_AUTO_TEST_CASE(test_pinmodel)
  {
    pinocchio::Model model;
    pinocchio::buildModels::humanoidRandom(model, true);

    std::cout << "test_pinmodel" << '\n';
    using M = lienlp::PinocchioGroup<double>;
    M space(model);
    M::Vec_t x0(model.nq);
    x0.noalias() = pinocchio::neutral(model);
    M::Vec_t v(model.nv);
    v.setRandom();
    std::cout << v << std::endl;
    M::Vec_t xout(model.nq);
    space.integrate(x0, v, xout);
    std::cout << "xout:" << xout << std::endl;

    // M::Vec_t x1 = pinocchio::randomConfiguration(model);
    // M::Vec_t out;
    // out.setZero(model.nv);
    // space.diff(x0, x1, out);
    // std::cout << out << std::endl;
  }
// #endif

BOOST_AUTO_TEST_SUITE_END()
