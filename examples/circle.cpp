/**
 * Optimize a quadratic function on a circle, or on a disk.
 *
 */
#include "proxsuite-nlp/cost-function.hpp"
#include "proxsuite-nlp/pdal.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"
#include "proxsuite-nlp/modelling/costs/squared-distance.hpp"
#include "proxsuite-nlp/modelling/costs/quadratic-residual.hpp"
#include "proxsuite-nlp/modelling/constraints/negative-orthant.hpp"
#include "proxsuite-nlp/prox-solver.hpp"

#include "example-base.hpp"

using namespace proxsuite::nlp;
using Manifold = VectorSpaceTpl<double>;
using Problem = ProblemTpl<double>;

int main() {
  constexpr int dim = 2;
  Manifold space{dim};
  const int nx = space.nx();
  Manifold::VectorXs p0(nx); // target
  p0 << -.4, .7;
  fmt::print("  |p0| = {}", p0.norm());
  Manifold::VectorXs p1(nx);
  p1 << 1., 0.5;
  fmt::print("{} << p0\n", p0);
  fmt::print("{} << p1\n", p1);

  const int ndx = space.ndx();
  Manifold::VectorXs d(ndx);
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

  auto cost_fun =
      std::make_shared<QuadraticDistanceCostTpl<double>>(space, p0, weights);
  const auto &cf = *cost_fun;
  fmt::print("cost: {}\n", cf.call(p1));
  fmt::print("grad: {}\n", cf.computeGradient(p1));
  fmt::print("hess: {}\n", cf.computeHessian(p1));

  using DistResType = ManifoldDifferenceToPoint<double>;
  auto resptr = std::make_shared<DistResType>(space, space.neutral());
  const auto &residual = *resptr;
  fmt::print("residual val @ p0: {}\n", residual(p0).transpose());
  fmt::print("residual val @ p1: {}\n", residual(p1).transpose());
  fmt::print("residual Jac: {}\n", residual.computeJacobian(p1));

  /// DEFINE A PROBLEM

  double radius_ = .6;
  double radius_sq = radius_ * radius_;
  Problem::MatrixXs w2(ndx, ndx);
  w2.setIdentity();
  w2 *= 2;

  auto residualCirclePtr = std::make_shared<QuadraticResidualCostTpl<double>>(
      resptr, w2, -radius_sq);

  using Inequality = NegativeOrthantTpl<double>;
  using CstrType = ConstraintObjectTpl<double>;
  CstrType cstr1(residualCirclePtr, Inequality{});
  std::vector<CstrType> cstrs{cstr1};
  Problem prob(space, cost_fun, cstrs);

  /// Test out merit functions

  Problem::VectorXs grad(space.ndx());
  grad.setZero();
  Problem::MatrixXs hess(space.ndx(), space.ndx());
  hess.setZero();

  // PDAL FUNCTION
  fmt::print("  LAGR FUNC TEST\n");

  Problem::VectorXs lams_data(prob.getTotalConstraintDim());
  Problem::VectorOfRef lams;
  helpers::allocateMultipliersOrResiduals(prob, lams_data, lams);

  fmt::print("Allocated {:d} multipliers | 1st mul = {}\n", lams.size(),
             lams[0]);

  // gradient of merit fun

  ProxNLPSolverTpl<double> solver(prob);
  solver.verbose = VERYVERBOSE;
  solver.setPenalty(1. / 50);

  auto lams0 = lams;
  solver.ldlt_choice_ = LDLTChoice::DENSE;
  solver.setup();
  solver.solve(p1, lams0);
  auto &results = *solver.results_;
  fmt::print("Results: {}\n", results);
  fmt::print("Output point: {}\n", results.x_opt.transpose());
  fmt::print("Target point was {}\n", p0.transpose());

  return 0;
}
