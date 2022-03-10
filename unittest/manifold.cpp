#include "lienlp/manifold-base.hpp"
#include "lienlp/spaces/pinocchio-groups.hpp"
#include "lienlp/spaces/multibody.hpp"

#include <pinocchio/parsers/sample-models.hpp>
#include <pinocchio/multibody/liegroup/vector-space.hpp>

#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>


using namespace lienlp;


BOOST_AUTO_TEST_SUITE(manifold)

BOOST_AUTO_TEST_CASE(test_lg_vecspace)
{
  std::cout << "test RN" << '\n';
  const int N = 4;
  using Vs = pinocchio::VectorSpaceOperationTpl<N, double>;
  const Vs lg;
  PinocchioLieGroup<Vs> space(lg);
  Vs::ConfigVector_t x0(space.nx());
  x0.setRandom();
  Vs::TangentVector_t v0(space.ndx());
  v0.setZero();
  Vs::TangentVector_t v1(space.ndx());
  v1.setRandom();

  std::cout << x0 << "<- x0\n";

  auto x1 = space.integrate(x0, v0);
  std::cout << x1 << std::endl;
  BOOST_CHECK(x1.isApprox(x0));

}


/// The tangent bundle of the SO2 Lie group.
BOOST_AUTO_TEST_CASE(test_so2_tangent)
{
  BOOST_TEST_MESSAGE("Starting T(SO2) test");
  using _SO2 = pinocchio::SpecialOrthogonalOperationTpl<2, double>;
  using SO2_wrap = PinocchioLieGroup<_SO2>;
  _SO2 lg_;
  SO2_wrap base_space(lg_);
  using TSO2 = TangentBundle<SO2_wrap>;
  TSO2 tspace(base_space);

  BOOST_TEST_MESSAGE("Checking bundle dimension");
  // tangent bundle dim should be 3.
  BOOST_CHECK_EQUAL(tspace.nx(), 3);

  auto x0 = tspace.zero();
  BOOST_CHECK(x0.isApprox(Eigen::Vector3d(1., 0., 0.)));
  auto x1 = tspace.rand();
  
  BOOST_TEST_MESSAGE(" testing diff");
  TSO2::TangentVec_t dx0(2);
  tspace.difference(x0, x1, dx0);
  std::cout << dx0 << " << dx0" << std::endl;


  BOOST_TEST_MESSAGE(" diff Jacobians");
  TSO2::Jac_t J0, J1;

  tspace.Jdifference(x0, x1, J0, 0);
  std::cout << "J0:\n" << J0 << std::endl;

  tspace.Jdifference(x0, x1, J1, 1);
  std::cout << "J1:\n" << J1 << std::endl;

  BOOST_CHECK(J0.isApprox(-TSO2::Jac_t::Identity(2, 2)));
  BOOST_CHECK(J1.isApprox( TSO2::Jac_t::Identity(2, 2)));

  J0.setZero();
  J1.setZero();

  // INTEGRATION OP
  BOOST_TEST_MESSAGE(" testing integration");
  TSO2::Point_t x1_new;
  tspace.integrate(x0, dx0, x1_new);
  BOOST_CHECK(x1_new.isApprox(x1));

  BOOST_TEST_MESSAGE(" integrate jacobians");

  tspace.Jintegrate(x0, dx0, J0, 0);
  std::cout << J0 << " << J0\n";

  tspace.Jintegrate(x0, dx0, J1, 1);
  std::cout << J1 << " << J1"   << std::endl;

}


// #ifdef WITH_PINOCCHIO_SUPPORT
BOOST_AUTO_TEST_CASE(test_pinmodel)
{
  BOOST_TEST_MESSAGE("Starting");

  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model, true);

  using Q_t = MultibodyConfiguration<double>;
  using Vec_t = Q_t::Point_t;
  Q_t space(model);

  Vec_t x0 = pinocchio::neutral(model);
  Vec_t d(model.nv);
  d.setRandom();

  Vec_t xout(model.nq);
  space.integrate(x0, d, xout);
  auto xout2 = pinocchio::integrate(model, x0, d);
  BOOST_CHECK(xout.isApprox(xout2));
  std::cout << "  integrate OK\n";

  Vec_t x1;
  d.setZero();
  x1 = pinocchio::randomConfiguration(model);
  space.difference(x0, x0, d);
  BOOST_CHECK(d.isZero());
  std::cout << "  diff OK\n";

  space.difference(x0, x1, d);
  BOOST_CHECK(d.isApprox(pinocchio::difference(model, x0, x1)));
  std::cout << "  diff OK\n";
}
// #endif

/// Test the tangent bundle specialization on rigid multibodies.
BOOST_AUTO_TEST_CASE(test_tangentbundle_multibody)
{
  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model, true);

  using M_t = StateMultibody<double>;

  // MultibodyConfiguration<double> config_space(model);
  // M_t space(config_space);
  M_t space(model);

  auto x0 = space.zero();
  auto x1 = space.rand();

}


BOOST_AUTO_TEST_SUITE_END()
