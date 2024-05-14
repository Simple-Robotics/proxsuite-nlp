/**
 * Optimize a quadratic function on a circle, or on a disk.
 *
 */
#include "proxsuite-nlp/modelling/spaces/pinocchio-groups.hpp"
#include "proxsuite-nlp/modelling/constraints/equality-constraint.hpp"
#include "proxsuite-nlp/prox-solver.hpp"

#include <pinocchio/multibody/liegroup/special-orthogonal.hpp>

using SO2 = pinocchio::SpecialOrthogonalOperationTpl<2, double>;

using namespace proxsuite::nlp;
using Manifold = PinocchioLieGroup<SO2>;
using Problem = ProblemTpl<double>;

int main() {
  auto space_ = std::make_shared<Manifold>();
  const Manifold &space = *space_;
  Manifold::PointType p0 = space.rand(); // target
  Manifold::PointType p1 = space.rand();
  Manifold::PointType neut = space.neutral();
  fmt::print("{} << p0\n", p0);
  fmt::print("{} << p1\n", p1);

  const int ndx = space.ndx();
  Manifold::MatrixXs J0(ndx, ndx), J1(ndx, ndx);
  J0.setZero();
  J1.setZero();

  space.Jdifference(p0, p1, J0, 0);
  space.Jdifference(p0, p1, J1, 1);
  fmt::print("J0 = {}\n", J0);
  fmt::print("J1 = {}\n", J1);

  Manifold::MatrixXs weights(ndx, ndx);
  weights.setIdentity();

  ManifoldDifferenceToPoint<double> residual(space_, p0);
  fmt::print("residual val: {}\n", residual(p1));
  fmt::print("residual Jac: {}\n", residual.computeJacobian(p1));
  auto resptr = std::make_shared<ManifoldDifferenceToPoint<double>>(residual);

  auto cost_fun =
      std::make_shared<QuadraticResidualCostTpl<double>>(resptr, weights);
  const auto &cf = *cost_fun;
  fmt::print("cost: {}\n", cf(p1));
  fmt::print("grad: {}\n", cf.computeGradient(p1));
  fmt::print("hess: {}\n", cf.computeHessian(p1));

  /// DEFINE A PROBLEM

  auto eq_set = std::make_shared<EqualityConstraintTpl<double>>();
  std::vector<Problem::ConstraintObject> cstrs;
  cstrs.emplace_back(resptr, eq_set);
  Problem prob(space_, cost_fun, cstrs);

  /// Test out merit functions

  Problem::VectorXs grad(ndx);
  grad.setZero();
  Problem::MatrixXs hess(space.ndx(), space.ndx());
  hess.setZero();

  // PDAL FUNCTION
  fmt::print("pdAL function test\n");

  Problem::VectorXs lams_data;
  Problem::VectorOfRef lams;
  helpers::allocateMultipliersOrResiduals(prob, lams_data, lams);

  fmt::print("Allocated {:d} multipliers\n"
             "1st mul = {}\n",
             lams.size(), lams[0]);

  ProxNLPSolverTpl<double> solver(prob, 0.01);
  solver.setup();
  solver.solve(space.rand());

  auto &rs = *solver.results_;
  fmt::print("Results: {}\n", rs);

  return 0;
}
