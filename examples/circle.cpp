/**
 * Optimize a quadratic function on a circle, or on a disk.
 * 
 */
#include "lienlp/cost-function.hpp"
#include "lienlp/merit-function-base.hpp"
#include "lienlp/meritfuncs/pdal.hpp"
#include "lienlp/modelling/spaces/pinocchio-groups.hpp"
#include "lienlp/modelling/costs/squared-distance.hpp"
#include "lienlp/modelling/residuals/quadratic-residual.hpp"
#include "lienlp/solver-base.hpp"

#include <pinocchio/multibody/liegroup/special-orthogonal.hpp>


#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ostream.h>
#include <Eigen/Core>


using Vs = pinocchio::VectorSpaceOperationTpl<2, double>;

using fmt::format;

using namespace lienlp;
using Man = PinocchioLieGroup<Vs>;
using Prob_t = Problem<double>;

int main()
{
  Man space;
  auto lg = space.m_lg;
  Man::Point_t p0 = lg.random();  // target
  p0.normalize();
  p0 << -.4, .7;
  Man::Point_t p1;
  p1 << 1., 0.5;
  fmt::print("{} << p0\n", p0);
  fmt::print("{} << p1\n", p1);

  Man::TangentVec_t d;
  space.difference(p0, p1, d);
  Man::Jac_t J0, J1;
  space.Jdifference<0>(p0, p1, J0);
  space.Jdifference<1>(p0, p1, J1);
  fmt::print("{} << p1 (-) p0\n", d);
  fmt::print("J0 = {}\n", J0);
  fmt::print("J1 = {}\n", J1);

  Man::Jac_t weights;
  weights.setIdentity();

  StateResidual<Man> residual(space, p0);
  fmt::print("residual val @ p0: {}\n", residual(p0).transpose());
  fmt::print("residual val @ p1: {}\n", residual(p1).transpose());
  fmt::print("residual Jac: {}\n", residual.computeJacobian(p1));
  auto resptr = std::make_shared<decltype(residual)>(residual);

  QuadraticResidualCost<double> cf(resptr, weights);
  // auto cf = WeightedSquareDistanceCost<Man>(space, p0, weights);
  fmt::print("cost: {}\n", cf(p1));
  fmt::print("grad: {}\n", cf.computeGradient(p1));
  fmt::print("hess: {}\n", cf.computeHessian(p1));

  /// DEFINE A PROBLEM

  // Prob_t::CstrPtr cstr1(new Prob_t::Equality_t(residual));
  QuadraticResidualFunctor<Man> residualCircle(space, 1., space.zero());
  Prob_t::Equality_t cstr1(residualCircle);
  fmt::print("  Cstr eval(p0): {}\n", cstr1(p0));
  fmt::print("  Cstr eval(p1): {}\n", cstr1(p1));
  fmt::print("  Constraint dimension: {:d}\n", cstr1.nr());

  std::vector<Prob_t::CstrPtr> cstrs;
  cstrs.push_back(std::make_shared<Prob_t::Equality_t>(residualCircle));
  shared_ptr<Prob_t> prob(new Prob_t(cf, cstrs));

  /// Test out merit functions

  Prob_t::VectorXs grad(space.ndx());
  EvalObjective<double> merit_fun(prob);
  fmt::print("eval merit fun:  M={}\n", merit_fun(p1));
  merit_fun.computeGradient(p0, grad);
  fmt::print("eval merit grad: ∇M={}\n", grad);


  // PDAL FUNCTION
  fmt::print("  LAGR FUNC TEST\n");

  PDALFunction<double> pdmerit(prob);
  auto lagr = pdmerit.m_lagr;
  Prob_t::VectorOfVectors lams;
  Prob_t::allocateMultipliers(*prob, lams);

  fmt::print("Allocated {:d} multipliers | 1st mul = {}\n",
             lams.size(), lams[0]);

  // lagrangian
  fmt::print("\tL(p0) = {}\n", lagr(p0, lams));
  fmt::print("\tL(p1) = {}\n", lagr(p1, lams));
  lagr.computeGradient(p0, lams, grad);
  fmt::print("\tgradL(p0) = {}\n", grad);
  lagr.computeGradient(p1, lams, grad);
  fmt::print("\tgradL(p1) = {}\n", grad);

  Prob_t::MatrixXs hess(space.ndx(), space.ndx());
  lagr.computeHessian(p0, lams, hess);
  fmt::print("\tHLag(p0) = {}\n", hess);
  lagr.computeHessian(p1, lams, hess);
  fmt::print("\tHLag(p1) = {}\n", hess);

  // merit function
  fmt::print("  PDAL FUNC TEST\n");
  fmt::print("\tpdmerit(p0) = {}\n", pdmerit(p0, lams, lams));
  fmt::print("\tpdmerit(p1) = {}\n", pdmerit(p1, lams, lams));

  // gradient of merit fun
  pdmerit.computeGradient(p0, lams, lams, grad);
  fmt::print("\tgradM(p0) {}\n", grad);
  pdmerit.computeGradient(p1, lams, lams, grad);
  fmt::print("\tgradM(p1) {}\n", grad);

  SWorkspace<double> workspace(space.nx(), space.ndx(), *prob);
  SResults<double> results(space.nx(), *prob);

  Solver<Man> solver(space, prob);
  solver.setPenalty(1. / 50);
  solver.useGaussNewton = true;

  auto lams0 = lams;
  fmt::print(fmt::fg(fmt::color::green), "[CALLING SOLVER]\n");
  solver.solve(workspace, results, p1, lams0);
  fmt::print("Results: {}\n", results);
  fmt::print("Target point was {}\n", p0.transpose());

  return 0;
}
