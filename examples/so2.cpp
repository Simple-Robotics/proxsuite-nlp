/**
 * Optimize a quadratic function on a circle, or on a disk.
 * 
 */
#include "lienlp/cost-function.hpp"
#include "lienlp/merit-function-base.hpp"
#include "lienlp/meritfuncs/pdal.hpp"
#include "lienlp/modelling/spaces/pinocchio-groups.hpp"
#include "lienlp/modelling/costs/squared-distance.hpp"

#include <pinocchio/multibody/liegroup/special-orthogonal.hpp>
#include "example-base.hpp"


using SO2 = pinocchio::SpecialOrthogonalOperationTpl<2, double>;

using fmt::format;

using namespace lienlp;
using Manifold = PinocchioLieGroup<SO2>;
using Problem = ProblemTpl<double>;

int main()
{
  Manifold space;
  SO2 lg = space.m_lg;
  Manifold::PointType neut = lg.neutral();
  Manifold::PointType p0 = lg.random();  // target
  Manifold::PointType p1 = lg.random();
  fmt::print("{} << p0\n", p0);
  fmt::print("{} << p1\n", p1);
  Manifold::TangentVectorType th0(1), th1(1);
  th0.setZero();
  th1.setZero();
  space.difference(neut, p0, th0);
  space.difference(neut, p1, th1);

  fmt::print("Angles:\n\tth0={}\n\tth1={}\n", th0, th1);

  const int ndx = space.ndx();
  Manifold::TangentVectorType d(ndx);
  d.setZero();
  Manifold::JacobianType J0(ndx, ndx), J1(ndx, ndx);
  J0.setZero();
  J1.setZero();

  space.difference(p0, p1, d);
  space.Jdifference(p0, p1, J0, 0);
  space.Jdifference(p0, p1, J1, 1);
  fmt::print("{} << p1 (-) p0\n", d);
  fmt::print("J0 = {}\n", J0);
  fmt::print("J1 = {}\n", J1);

  Manifold::JacobianType weights(ndx, ndx);
  weights.setIdentity();

  ManifoldDifferenceToPoint<double> residual(space, p0);
  fmt::print("residual val: {}\n", residual(p1));
  fmt::print("residual Jac: {}\n", residual.computeJacobian(p1));
  auto resptr = std::make_shared<ManifoldDifferenceToPoint<double>>(residual);

  QuadraticResidualCost<double> cf(resptr, weights);
  fmt::print("cost: {}\n", cf(p1));
  fmt::print("grad: {}\n", cf.computeGradient(p1));
  fmt::print("hess: {}\n", cf.computeHessian(p1));

  /// DEFINE A PROBLEM

  Problem::ConstraintPtr cstr1(new Problem::EqualityType(residual));
  std::vector<Problem::ConstraintPtr> cstrs;
  cstrs.push_back(cstr1);
  shared_ptr<Problem> prob(new Problem(cf, cstrs));
  fmt::print("\tConstraint dimension: {:d}\n", prob->getConstraint(0)->nr());

  /// Test out merit functions

  Problem::VectorXs grad(ndx);
  grad.setZero();
  Problem::MatrixXs hess(space.ndx(), space.ndx());
  hess.setZero();

  EvalObjective<double> merit_fun(prob);
  fmt::print("eval merit fun :  M={}\n", merit_fun(p1));
  merit_fun.computeGradient(p0, grad);
  fmt::print("eval merit grad: ∇M={}\n", grad);


  // PDAL FUNCTION
  fmt::print("  LAGR FUNC TEST\n");

  PDALFunction<double> pdmerit(prob);
  auto lagr = pdmerit.m_lagr;
  Problem::VectorXs lams_data;
  Problem::VectorOfRef lams;
  helpers::allocateMultipliersOrResiduals(*prob, lams_data, lams);

  fmt::print("Allocated {:d} multipliers\n"
             "1st mul = {}\n", lams.size(), lams[0]);

  // lagrangian
  fmt::print("\tL(p0) = {}\n", lagr(p0, lams));
  fmt::print("\tL(p1) = {}\n", lagr(p1, lams));
  lagr.computeGradient(p0, lams, grad);
  fmt::print("\tgradL(p0) = {}\n", grad);
  lagr.computeGradient(p1, lams, grad);
  fmt::print("\tgradL(p1) = {}\n", grad);

  lagr.computeHessian(p0, lams, hess);
  fmt::print("\tHLag(p0) = {}\n", hess);
  lagr.computeHessian(p1, lams, hess);
  fmt::print("\tHLag(p1) = {}\n", hess);

  // merit function
  fmt::print("  PDAL FUNC TEST\n");
  fmt::print("\tpdmerit(p0) = {}\n", pdmerit(p0, lams, lams));
  fmt::print("\tpdmerit(p1) = {}\n", pdmerit(p1, lams, lams));
  pdmerit.computeHessian(p0, lams, lams, hess);
  fmt::print("\tHmerit(p0) = {}\n", hess);
  pdmerit.computeHessian(p1, lams, lams, hess);
  fmt::print("\tHmerit (p1) = {}\n", hess);

  // gradient of merit fun
  pdmerit.computeGradient(p0, lams, lams, grad);
  fmt::print("\tgradM(p0) {}\n", grad);
  pdmerit.computeGradient(p1, lams, lams, grad);
  fmt::print("\tgradM(p1) {}\n", grad);

  return 0;
}
