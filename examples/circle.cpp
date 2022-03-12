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


#include <fmt/format.h>
#include <fmt/ostream.h>
#include <Eigen/Core>


using SO2 = pinocchio::SpecialOrthogonalOperationTpl<2, double>;
using Man = lienlp::PinocchioLieGroup<SO2>;

using fmt::format;

using namespace lienlp;

int main()
{
  SO2 lg;
  Man space(lg);
  Man::Point_t neut = lg.neutral();
  Man::Point_t p0 = lg.random();  // target
  Man::Point_t p1 = lg.random();
  fmt::print("{} << p0\n", p0);
  fmt::print("{} << p1\n", p1);
  Man::TangentVec_t th0(1), th1(1);
  space.difference(neut, p0, th0);
  space.difference(neut, p1, th1);

  fmt::print("Angles:\n\tth0={}\n\tth1={}\n", th0, th1);

  Man::TangentVec_t d;
  space.difference(p0, p1, d);
  Man::Jac_t J0, J1;
  space.Jdifference(p0, p1, J0, 0);
  space.Jdifference(p0, p1, J1, 1);
  fmt::print("{} << p1 (-) p0\n", d);
  fmt::print("J0 = {}\n", J0);
  fmt::print("J1 = {}\n", J1);

  Eigen::Matrix<double, Man::NV, Man::NV> weights;
  weights.setIdentity();

  StateResidual<Man> residual(&space, p0);
  fmt::print("residual val: {}\n", residual(p1));
  fmt::print("residual Jac: {}\n", residual.jacobian(p1));

  auto cf = QuadResidualCost<double>(&space, &residual, weights);
  // auto cf = WeightedSquareDistanceCost<Man>(space, p0, weights);
  fmt::print("cost: {}\n", cf(p1));
  fmt::print("grad: {}\n", cf.gradient(p1));
  fmt::print("hess: {}\n", cf.hessian(p1));

  /// DEFINE A PROBLEM

  using Prob_t = Problem<double>;
  Prob_t::CstrPtr cstr1(new Prob_t::Equality_t(residual, 1));
  std::vector<Prob_t::CstrPtr> cstrs;
  cstrs.push_back(cstr1);
  Prob_t prob(cf, cstrs);
  fmt::print("\tConstraint dimension: {:d}\n", prob.getCstr(0)->getDim());

  /// Test out merit functions

  fmt::print("  MERIT FUNC TEST\n");
  EvalObjective<double> merit_fun(&prob);
  fmt::print("eval merit fun:  M={}\n", merit_fun(p1));
  fmt::print("eval merit grad: ∇M={}\n", merit_fun.gradient(p1));


  // PDAL FUNCTION
  fmt::print("  LAGR FUNC TEST\n");

  PDALFunction<double> pdmerit(&prob);
  auto lagr = pdmerit.m_lagr;
  Prob_t::VectorList lams;
  prob.allocateMultipliers(lams);
  fmt::print("Allocated {:d} multipliers\n"
             "1st mul = {}\n", lams.size(), lams[0]);

  // lagrangian
  Prob_t::VectorXs grad(space.ndx());
  fmt::print("\tL(p0) = {}\n", lagr(p0, lams));
  fmt::print("\tL(p1) = {}\n", lagr(p1, lams));
  lagr.gradient(p0, lams, grad);
  fmt::print("\tgradL(p0) = {}\n", grad);
  lagr.gradient(p1, lams, grad);
  fmt::print("\tgradL(p1) = {}\n", grad);

  fmt::print("  PDAL FUNC TEST\n");
  fmt::print("\tpdmerit(p0) = {}\n", pdmerit(p0, lams, lams));
  fmt::print("\tpdmerit(p1) = {}\n", pdmerit(p1, lams, lams));

  // gradient of merit fun
  pdmerit.gradient(p0, lams, lams, grad);
  fmt::print("\tgradM(p0) {}\n", grad);
  pdmerit.gradient(p1, lams, lams, grad);
  fmt::print("\tgradM(p1) {}\n", grad);

  return 0;
}
