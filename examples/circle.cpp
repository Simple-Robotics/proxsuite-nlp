#include "lienlp/cost-function.hpp"
#include "lienlp/merit-function-base.hpp"
#include "lienlp/spaces/pinocchio-groups.hpp"
#include "lienlp/costs/squared-distance.hpp"

#include <pinocchio/multibody/liegroup/special-orthogonal.hpp>


#include <Eigen/Core>
#include <iostream>
#include <math.h>


using SO2 = pinocchio::SpecialOrthogonalOperationTpl<2, double>;
using Man = lienlp::PinocchioLieGroup<SO2>;

using namespace lienlp;

int main()
{
  SO2 lg;
  Man space(lg);
  Man::Point_t neut = lg.neutral();
  Man::Point_t p0 = lg.random();  // target
  Man::Point_t p1 = lg.random();
  std::cout << p0 << " << p0\n";
  std::cout << p1 << " << p1\n";
  auto th0 = space.diff(neut, p0);
  auto th1 = space.diff(neut, p1);
  std::cout << "Angles:\n\t";
  std::cout << th0 << "  << th0\n\t";
  std::cout << th1 << "  << th1\n";

  std::cout << "norm:" << p0.transpose() * p0 << '\n';

  auto d = space.diff(p0, p1);
  Man::Jac_t J0, J1;
  space.Jdiff(p0, p1, J0, 0);
  space.Jdiff(p0, p1, J1, 1);
  std::cout << d << "  << p1 (-) p0\n";
  std::cout << J0 << " << J0\n";
  std::cout << J1 << " << J1\n";
  std::cout << space.diff(p0, p1) << "  << diff (out of place)\n";

  Eigen::Matrix<double, Man::NV, Man::NV> weights;
  weights.setIdentity();
  std::cout << "Weights:\n" << weights << '\n';

  using SR = StateResidual<Man>;
  SR residual(space, p0);
  std::cout << residual.m_manifold.diff(p0, p1) << "  << res eval\n";
  std::cout << residual.m_target << "  << res target\n\n";

  std::cout << "residual v: " << residual(p1) << '\n';
  std::cout << "residual J: " << residual.jacobian(p1) << '\n';

  auto cf = QuadResidualCost<SR>(space, residual, weights);
  // auto cf = WeightedSquareDistanceCost<Man>(space, p0, weights);
  std::cout << "cost: " << cf(p1) << '\n';
  std::cout << "grad: " << cf.gradient(p1) << '\n';
  std::cout << "hess: " << cf.hessian(p1) << '\n';

  // auto merit_fun = EvalObjective<Man>(cf);
  // std::cout << "eval merit fun:   M=" << merit_fun(p1)          << '\n';
  // std::cout << "eval merit grad: ∇M=" << merit_fun.gradient(p1) << '\n';

  return 0;
}
