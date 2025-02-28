#include <boost/test/unit_test.hpp>
#include "util.hpp"

#include "proxsuite-nlp/bfgs-strategy.hpp"

using namespace proxsuite::nlp;

BOOST_AUTO_TEST_CASE(test_inverse_hessian_update) {
  const long nx = 4;
  using BFGSStrategy_t = BFGSStrategy<Scalar>; // default to InverseHessian
  BFGSStrategy_t bfgs(nx);
  VectorXs x0 = VectorXs::Random(nx);
  VectorXs g0 = VectorXs::Random(nx);
  bfgs.init(x0, g0);

  VectorXs x;
  VectorXs g;
  VectorXs s;
  VectorXs y;

  bool is_psd = false;
  MatrixXs H;
  while (!is_psd) {
    x = VectorXs::Random(nx);
    g = VectorXs::Random(nx);
    s = x - x0;
    y = g - g0;
    if (s.dot(y) > 0) {
      bfgs.update(x, g, H);
      is_psd = bfgs.is_psd;
    }
  }
  Scalar rho = 1. / s.dot(y);
  MatrixXs H_inv_prev = MatrixXs::Identity(nx, nx);
  MatrixXs H_inv_expected =
      (MatrixXs::Identity(nx, nx) - rho * s * y.transpose()) * H_inv_prev *
          (MatrixXs::Identity(nx, nx) - rho * y * s.transpose()) +
      rho * s * s.transpose();
  BOOST_TEST_CHECK(H.isApprox(H_inv_expected, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_hessian_update) {
  const long nx = 4;
  using BFGSStrategy_t = BFGSStrategy<Scalar, BFGSType::Hessian>;
  BFGSStrategy_t bfgs(nx);
  VectorXs x0 = VectorXs::Random(nx);
  VectorXs g0 = VectorXs::Random(nx);
  bfgs.init(x0, g0);

  VectorXs x;
  VectorXs g;
  VectorXs s;
  VectorXs y;

  bool is_psd = false;
  MatrixXs H;
  while (!is_psd) {
    x = VectorXs::Random(nx);
    g = VectorXs::Random(nx);
    s = x - x0;
    y = g - g0;
    if (s.dot(y) > 0) {
      bfgs.update(x, g, H);
      is_psd = bfgs.is_psd;
    }
  }

  Scalar rho = 1. / s.dot(y);
  MatrixXs H_prev = MatrixXs::Identity(nx, nx);
  MatrixXs H_expected =
      (MatrixXs::Identity(nx, nx) - rho * y * s.transpose()) * H_prev *
          (MatrixXs::Identity(nx, nx) - rho * s * y.transpose()) +
      rho * y * y.transpose();
  BOOST_TEST_CHECK(H.isApprox(H_expected, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_bfgs_inverse_hessian) {
  const long nx = 4;
  using BFGSStrategy_t = BFGSStrategy<double, BFGSType::InverseHessian>;
  BFGSStrategy_t bfgs(nx);

  // random quadratic function with positive definite Hessian
  Eigen::MatrixXd H = Eigen::MatrixXd::Random(nx, nx);
  H = H.transpose() * H; // make it symmetric and positive definite
  Eigen::MatrixXd H_inv = H.inverse();
  Eigen::VectorXd b = Eigen::VectorXd::Random(nx);

  Eigen::VectorXd x0 = Eigen::VectorXd::Random(nx);
  Eigen::VectorXd g0 = H * x0 - b;
  bfgs.init(x0, g0);

  MatrixXs H_inv_approx;
  for (int i = 0; i < 1000; ++i) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(nx);
    Eigen::VectorXd g = H * x - b; // gradient
    bfgs.update(x, g, H_inv_approx);

    Eigen::MatrixXd H_inv_err = H_inv - H_inv_approx;
    double err = H_inv_err.norm();
    if (err < 1e-5) {
      break;
    }
  }

  BOOST_TEST_CHECK(bfgs.M.isApprox(H_inv, 1e-5));
}

BOOST_AUTO_TEST_CASE(test_bfgs_hessian) {
  const long nx = 4;
  using BFGSStrategy_t = BFGSStrategy<double, BFGSType::Hessian>;
  BFGSStrategy_t bfgs(nx);

  // random quadratic function with positive definite Hessian
  Eigen::MatrixXd H = Eigen::MatrixXd::Random(nx, nx);
  H = H.transpose() * H; // make it symmetric and positive definite
  Eigen::VectorXd b = Eigen::VectorXd::Random(nx);

  Eigen::VectorXd x0 = Eigen::VectorXd::Random(nx);
  Eigen::VectorXd g0 = H * x0 - b;
  bfgs.init(x0, g0);

  MatrixXs H_approx;
  for (int i = 0; i < 1000; ++i) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(nx);
    Eigen::VectorXd g = H * x - b; // gradient
    bfgs.update(x, g, H_approx);

    Eigen::MatrixXd H_err = H - H_approx;
    double err = H_err.norm();
    if (err < 1e-5) {
      break;
    }
  }

  BOOST_TEST_CHECK(bfgs.M.isApprox(H, 1e-5));
}
