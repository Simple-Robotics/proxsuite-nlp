/**
 * Optimize a quadratic function on a circle, or on a disk.
 * 
 */
#include "lienlp/cost-function.hpp"
#include "lienlp/merit-function-base.hpp"
#include "lienlp/meritfuncs/pdal.hpp"
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
  auto th0 = space.difference(neut, p0);
  auto th1 = space.difference(neut, p1);
  std::cout << "Angles:\n\t";
  std::cout << th0 << "  << th0\n\t";
  std::cout << th1 << "  << th1\n";

  Man::TangentVec_t d;
  space.difference(p0, p1, d);
  Man::Jac_t J0, J1;
  space.Jdifference(p0, p1, J0, 0);
  space.Jdifference(p0, p1, J1, 1);
  std::cout << d << "  << p1 (-) p0\n";
  std::cout << J0 << " << J0\n";
  std::cout << J1 << " << J1\n";
  std::cout << space.difference(p0, p1) << "  << diff (out of place)\n";

  Eigen::Matrix<double, Man::NV, Man::NV> weights;
  weights.setIdentity();

  StateResidual<Man> residual(&space, p0);
  std::cout << residual.m_manifold->difference(p0, p1) << "  << res eval\n";
  std::cout << residual.m_target << "  << res target\n\n";

  std::cout << " residual val: " << residual(p1) << '\n';
  std::cout << " residual Jac: " << residual.jacobian(p1) << '\n';

  auto cf = QuadResidualCost<double>(&space, &residual, weights);
  // auto cf = WeightedSquareDistanceCost<Man>(space, p0, weights);
  std::cout << " cost: " << cf(p1) << '\n';
  std::cout << " grad: " << cf.gradient(p1) << '\n';
  std::cout << " hess: " << cf.hessian(p1) << '\n';

  /// DEFINE A PROBLEM

  using Prob_t = Problem<double>;
  Prob_t::Equality_t cstr1(residual, 1);
  std::vector<Prob_t::Equality_t*> eq_cstrs;
  eq_cstrs.push_back(&cstr1);
  Prob_t prob(cf, eq_cstrs);
  std::cout << "   Constraint dimension:" << prob.getEqCs(0)->getDim() << '\n';

  /// Test out merit functions

  std::cout << "  MERIT FUNC TEST\n";
  EvalObjective<double> merit_fun(&prob);
  std::cout << "eval merit fun:   M=" << merit_fun(p1)          << '\n';
  // std::cout << "eval merit grad: ∇M=" << merit_fun.gradient(p1) << '\n';


  // PDAL FUNCTION

  PDALFunction<double> pdmerit(&prob);
  Prob_t::VectorList lams;
  prob.allocateMultipliers(lams);
  double value = pdmerit(p0, lams, lams);
  std::cout << " pdmerit(x0) " << value;

  value = pdmerit(p1, lams, lams);
  std::cout << " pdmerit(x1) " << value;
  std::cout << std::endl;

  return 0;
}
