#include "proxnlp/manifold-base.hpp"

#include "proxnlp/modelling/spaces/vector-space.hpp"
#include "proxnlp/modelling/spaces/cartesian-product.hpp"

#ifdef WITH_PINOCCHIO
#include <pinocchio/parsers/sample-models.hpp>
#include "proxnlp/modelling/spaces/pinocchio-groups.hpp"
#include "proxnlp/modelling/spaces/multibody.hpp"
#endif

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(manifold)

using namespace proxnlp;
using Manifold = ManifoldAbstractTpl<double>;

BOOST_AUTO_TEST_CASE(test_vectorspace) {
  constexpr int N1 = 3;
  VectorSpaceTpl<double, N1> space1;

  auto x0 = space1.neutral();
  auto x1 = space1.rand();

  BOOST_CHECK_EQUAL(N1, x0.size());
  BOOST_CHECK_EQUAL(N1, x1.size());

  BOOST_CHECK((x0 + x1).isApprox(x1));

  constexpr int N2 = 35;

  VectorSpaceTpl<double> space2(N2);
  x0 = space2.neutral();
  x1 = space2.rand();

  BOOST_CHECK(x0.isApprox(Eigen::VectorXd::Zero(35)));

  CartesianProductTpl<double> prod1(space1, space2);
  BOOST_CHECK_EQUAL(prod1.nx(), N1 + N2);
  x0 = prod1.neutral();
  BOOST_CHECK_EQUAL(x0.size(), N1 + N2);
  x0 = prod1.rand();

  // test copy constructor
  VectorSpaceTpl<double, N1> space1_copy(space1);
  VectorSpaceTpl<double> space2_copy(space2);
  shared_ptr<Manifold> space2_ptr =
      std::make_shared<VectorSpaceTpl<double>>(space2);

  CartesianProductTpl<double> prod2 = space2_ptr * space2_ptr;
  x1 = prod2.rand();
}

#ifdef WITH_PINOCCHIO

BOOST_AUTO_TEST_CASE(test_lg_vecspace) {
  const int N = 4;
  using Vs = pinocchio::VectorSpaceOperationTpl<N, double>;
  PinocchioLieGroup<Vs> space;
  Vs::ConfigVector_t x0(space.nx());
  x0.setRandom();
  Vs::TangentVector_t v0(space.ndx());
  v0.setZero();
  Vs::TangentVector_t v1(space.ndx());
  v1.setRandom();

  auto x1 = space.integrate(x0, v0);
  BOOST_CHECK(x1.isApprox(x0));

  auto mid = space.interpolate(x0, x1, 0.5);
  BOOST_CHECK(mid.isApprox(0.5 * (x0 + x1)));

  // test copy ctor
  PinocchioLieGroup<Vs> space_copy(space);
}

/// The tangent bundle of the SO2 Lie group.
BOOST_AUTO_TEST_CASE(test_so2_tangent) {
  BOOST_TEST_MESSAGE("Starting T(SO2) test");
  using _SO2 = pinocchio::SpecialOrthogonalOperationTpl<2, double>;
  using SO2 = PinocchioLieGroup<_SO2>;
  using TSO2 = TangentBundleTpl<SO2>;
  TSO2 tspace; // no arg constructor

  BOOST_TEST_MESSAGE("Checking bundle dimension");
  // tangent bundle dim should be 3.
  BOOST_CHECK_EQUAL(tspace.nx(), 3);

  auto x0 = tspace.neutral();
  BOOST_CHECK(x0.isApprox(Eigen::Vector3d(1., 0., 0.)));
  auto x1 = tspace.rand();

  const int ndx = tspace.ndx();
  BOOST_CHECK_EQUAL(ndx, 2);
  BOOST_TEST_MESSAGE(" testing diff");
  TSO2::TangentVectorType dx0(ndx);
  dx0.setZero();
  tspace.difference(x0, x1, dx0);

  BOOST_TEST_MESSAGE("Testing interpolate");
  auto mid = tspace.interpolate(x0, x1, 0.5);

  BOOST_TEST_MESSAGE(" diff Jacobians");
  TSO2::MatrixXs J0(ndx, ndx), J1(ndx, ndx);
  J0.setZero();
  J1.setZero();

  tspace.Jdifference(x0, x1, J0, 0);
  tspace.Jdifference(x0, x1, J1, 1);

  TSO2::MatrixXs id(2, 2);
  id.setIdentity();
  BOOST_CHECK(J0.isApprox(-id));
  BOOST_CHECK(J1.isApprox(id));

  // INTEGRATION OP
  BOOST_TEST_MESSAGE("Testing integration");
  TSO2::PointType x1_new(tspace.nx());
  tspace.integrate(x0, dx0, x1_new);
  BOOST_CHECK(x1_new.isApprox(x1));

  BOOST_TEST_MESSAGE("Integrate jacobians");

  tspace.Jintegrate(x0, dx0, J0, 0);
  tspace.Jintegrate(x0, dx0, J1, 1);

  tspace.JintegrateTransport(x0, dx0, J0, 0);
}

BOOST_AUTO_TEST_CASE(test_pinmodel) {
  BOOST_TEST_MESSAGE("Starting");

  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model, true);

  using Q_t = MultibodyConfiguration<double>;
  using Point_t = Q_t::PointType;
  Q_t space(model);

  Point_t x0 = pinocchio::neutral(model);
  Point_t d(model.nv);
  d.setRandom();

  Point_t xout(model.nq);
  space.integrate(x0, d, xout);
  auto xout2 = pinocchio::integrate(model, x0, d);
  BOOST_CHECK(xout.isApprox(xout2));

  Point_t x1;
  d.setZero();
  x1 = pinocchio::randomConfiguration(model);
  space.difference(x0, x0, d);
  BOOST_CHECK(d.isZero());

  BOOST_TEST_MESSAGE("Testing interpolate ");
  auto mid = space.interpolate(x0, x1, 0.5);
  BOOST_CHECK(mid.isApprox(pinocchio::interpolate(model, x0, x1, 0.5)));

  space.difference(x0, x1, d);
  BOOST_CHECK(d.isApprox(pinocchio::difference(model, x0, x1)));
}
// #endif

/// Test the tangent bundle specialization on rigid multibodies.
BOOST_AUTO_TEST_CASE(test_tangentbundle_multibody) {
  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model, true);

  using Man = MultibodyPhaseSpace<double>;

  // MultibodyConfiguration<double> config_space(model);
  Man space(model);

  auto x0 = space.neutral();
  auto x1 = space.rand();
  auto dx0 = space.difference(x0, x1);
  auto x1_exp = space.integrate(x0, dx0);
  const int ndx = space.ndx();

  Eigen::VectorXd J0(ndx, ndx);
  J0.setRandom();
  space.JintegrateTransport(x0, dx0, J0, 0);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
