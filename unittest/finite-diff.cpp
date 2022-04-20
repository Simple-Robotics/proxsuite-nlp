#include "proxnlp/modelling/autodiff/finite-difference.hpp"
#include "proxnlp/modelling/spaces/pinocchio-groups.hpp"

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

using namespace proxnlp;

BOOST_AUTO_TEST_SUITE(finite_diff)

namespace pin = pinocchio;

using Vs = pin::VectorSpaceOperationTpl<-1, double>;

PROXNLP_FUNCTOR_TYPEDEFS(double)
static const double fd_eps = 1e-4;
static const double prec = std::sqrt(fd_eps);

struct MyFuncType : C1FunctionTpl<double>
{
  using C1FunctionTpl<double>::computeJacobian;
  const ManifoldAbstractTpl<double>& space;
  MyFuncType(const ManifoldAbstractTpl<double>& space)
    : C1FunctionTpl(space.nx(), space.ndx(), 1), space(space)
    , refpt(space.neutral())
    {}

  VectorXs refpt;

  ReturnType operator()(const ConstVectorRef& x) const
  {
    ReturnType out(1);
    VectorXs err = space.difference(x, refpt);
    out << 1. / 3. * std::pow(err.lpNorm<3>(), 3);
    return out;
  }

  void computeJacobian(const ConstVectorRef& x, MatrixRef Jout) const
  {
    auto err = space.difference(x, refpt);
    MatrixXs J(this->ndx(), this->ndx());
    J.setZero();
    space.Jdifference(x, refpt, J, 0);
    Jout.transpose() = J.transpose() * (err.array() * err.array().abs()).matrix();
  }
};

using autodiff::finite_difference_helper;

BOOST_AUTO_TEST_CASE(test1)
{
  int nx = 4;
  PinocchioLieGroup<Vs> space(nx);

  MyFuncType fun(space);
  using autodiff::TOC1;
  using autodiff::TOC2;
  using fd_type = finite_difference_helper<double, TOC1>;
  fd_type fdfun1(space, fun, fd_eps);
  VectorXs x0    = space.rand();
  MatrixXs J0_fd = fdfun1.computeJacobian(x0);
  MatrixXs J0    = fun.computeJacobian(x0);
  fmt::print("J0_fd\n{}\n", J0_fd);
  fmt::print("x0: {}\n", x0.transpose());

  fmt::print("should be:\n{}\n", fun.computeJacobian(x0));
  BOOST_CHECK(J0.isApprox(J0_fd, prec));

  VectorXs v0(fun.nr());
  v0.setOnes();

  finite_difference_helper<double, TOC2> fdfun2(space, fun, fd_eps);
  fmt::print("Hessian:\n{}\n", fdfun2.vectorHessianProduct(x0, v0));

}

BOOST_AUTO_TEST_CASE(test2)
{
  PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<2, double>> space;

  MyFuncType fun(space);
  finite_difference_helper<double> fdfun1(space, fun, fd_eps);

  auto x0    = space.rand();
  auto J0_fd = fdfun1.computeJacobian(x0);
  auto J0    = fun.computeJacobian(x0);
  fmt::print("x0: {}\n", x0.transpose());
  fmt::print("J0_fd\n{}\n", J0_fd);
  fmt::print("J0\n{}\n", J0);

  BOOST_CHECK(J0.isApprox(J0_fd, prec));

}
BOOST_AUTO_TEST_SUITE_END()

