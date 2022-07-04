/**
 * Optimize a quadratic function on a circle, or on a disk.
 *
 */
#include "proxnlp/cost-function.hpp"
#include "proxnlp/pdal.hpp"
#include "proxnlp/modelling/spaces/vector-space.hpp"
#include "proxnlp/modelling/costs/squared-distance.hpp"
#include "proxnlp/modelling/costs/quadratic-residual.hpp"
#include "proxnlp/modelling/constraints/negative-orthant.hpp"
#include "proxnlp/solver-base.hpp"

#include "example-base.hpp"

using namespace proxnlp;
using Manifold = VectorSpaceTpl<double>;
using Problem = ProblemTpl<double>;

int main() {
  constexpr int dim = 2;
  Manifold space(dim);
  const int nx = space.nx();
  Manifold::PointType p0(nx); // target
  p0 << -.4, .7;
  fmt::print("  |p0| = {}", p0.norm());
  Manifold::PointType p1(nx);
  p1 << 1., 0.5;
  fmt::print("{} << p0\n", p0);
  fmt::print("{} << p1\n", p1);

  const int ndx = space.ndx();
  Manifold::TangentVectorType d(ndx);
  space.difference(p0, p1, d);
  d.setZero();
  Manifold::MatrixXs J0(ndx, ndx), J1(ndx, ndx);
  J0.setZero();
  J1.setZero();
  space.Jdifference(p0, p1, J0, 0);
  space.Jdifference(p0, p1, J1, 1);
  fmt::print("{} << p1 (-) p0\n", d);
  fmt::print("J0 = {}\n", J0);
  fmt::print("J1 = {}\n", J1);

  Manifold::MatrixXs weights(ndx, ndx);
  weights.setIdentity();

  QuadraticDistanceCost<double> cf(space, p0, weights);
  fmt::print("cost: {}\n", cf.call(p1));
  fmt::print("grad: {}\n", cf.computeGradient(p1));
  fmt::print("hess: {}\n", cf.computeHessian(p1));

  ManifoldDifferenceToPoint<double> residual(space, space.neutral());
  fmt::print("residual val @ p0: {}\n", residual(p0).transpose());
  fmt::print("residual val @ p1: {}\n", residual(p1).transpose());
  fmt::print("residual Jac: {}\n", residual.computeJacobian(p1));
  auto resptr = std::make_shared<decltype(residual)>(residual);

  /// DEFINE A PROBLEM

  double radius_ = .6;
  double radius_sq = radius_ * radius_;
  Problem::MatrixXs w2(ndx, ndx);
  w2.setIdentity();
  w2 *= 2;

  const QuadraticResidualCost<double> residualCircle(resptr, w2, -radius_sq);

  using Ineq_t = NegativeOrthant<double>;
  auto cstr1 = std::make_shared<ConstraintObject<double>>(
      residualCircle, std::make_shared<Ineq_t>());
  std::vector<Problem::ConstraintPtr> cstrs{cstr1};
  auto prob = std::make_shared<Problem>(cf, cstrs);

  /// Test out merit functions

  Problem::VectorXs grad(space.ndx());
  grad.setZero();
  Problem::MatrixXs hess(space.ndx(), space.ndx());
  hess.setZero();

  EvalObjective<double> merit_fun(prob);
  fmt::print("eval merit fun:  M={}\n", merit_fun(p1));
  merit_fun.computeGradient(p0, grad);
  fmt::print("eval merit grad: ∇M={}\n", grad);

  // PDAL FUNCTION
  fmt::print("  LAGR FUNC TEST\n");

  PDALFunction<double> pdmerit(prob);
  LagrangianFunction<double> &lagr = pdmerit.m_lagr;
  Problem::VectorXs lams_data(prob->getTotalConstraintDim());
  Problem::VectorOfRef lams;
  helpers::allocateMultipliersOrResiduals(*prob, lams_data, lams);

  fmt::print("Allocated {:d} multipliers | 1st mul = {}\n", lams.size(),
             lams[0]);

  // lagrangian
  fmt::print("\tL(p0) = {}\n", lagr(p0, lams));
  fmt::print("\tL(p1) = {}\n", lagr(p1, lams));
  lagr.computeGradient(p0, lams, grad);
  fmt::print("\tgradL(p0) = {}\n", grad);
  lagr.computeGradient(p1, lams, grad);
  fmt::print("\tgradL(p1) = {}\n", grad);

  // merit function
  fmt::print("  PDAL FUNC TEST\n");
  fmt::print("\tpdmerit(p0) = {}\n", pdmerit(p0, lams, lams));
  fmt::print("\tpdmerit(p1) = {}\n", pdmerit(p1, lams, lams));

  // gradient of merit fun
  pdmerit.computeGradient(p0, lams, lams, grad);
  fmt::print("\tgradM(p0) {}\n", grad);
  pdmerit.computeGradient(p1, lams, lams, grad);
  fmt::print("\tgradM(p1) {}\n", grad);

  WorkspaceTpl<double> workspace(space.nx(), space.ndx(), *prob);
  ResultsTpl<double> results(space.nx(), *prob);

  SolverTpl<double> solver(space, prob);
  solver.verbose = VerboseLevel::VERY;
  solver.setPenalty(1. / 50);
  solver.use_gauss_newton = true;

  auto lams0 = lams;
  solver.solve(workspace, results, p1, lams0);
  fmt::print("Results: {}\n", results);
  fmt::print("Output point: {}\n", results.xOpt.transpose());
  fmt::print("Target point was {}\n", p0.transpose());

  return 0;
}
