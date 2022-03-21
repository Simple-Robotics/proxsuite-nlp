/** Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#include "lienlp/modelling/costs/squared-distance.hpp"
#include "lienlp/modelling/residuals/linear.hpp"
#include "lienlp/modelling/constraints/equality-constraint.hpp"
#include "lienlp/modelling/spaces/pinocchio-groups.hpp"
#include "lienlp/solver-base.hpp"

#include "example-base.hpp"


using Vs = pinocchio::VectorSpaceOperationTpl<2, double>;

using namespace lienlp;
using Man = PinocchioLieGroup<Vs>;
using Prob_t = Problem<double>;
using Equality_t = EqualityConstraint<double>;

int main(int argc, const char* argv[])
{
  Man space;
  Man::Point_t p0 = space.zero();
  Man::Point_t p1 = space.rand();

  Eigen::MatrixXd Qroot(2, 4);
  Qroot.setRandom();
  Eigen::Matrix2d Q_ = Qroot * Qroot.transpose();

  Eigen::MatrixXd A(1, 2);
  A.setRandom();
  Eigen::VectorXd b(1);
  b << 0.5;

  fmt::print("Linear residual:\n{} << Q\n", Q_);
  fmt::print("A {}\n", A);

  LinearResidual<double> res1(A, b);

  QuadDistanceCost<Man> cost(space, Q_);

  fmt::print("cost(p0)  {}\n", cost(p0));
  fmt::print("cost(p1)  {}\n", cost(p1));

  auto cstr1 = std::make_shared<Equality_t>(res1);
  std::vector<Prob_t::CstrPtr> cstrs_{
    cstr1,
    cstr1
    };

  auto prob = std::make_shared<Prob_t>(cost, cstrs_);

  using Solver_t = Solver<Man>;
  Solver_t::Workspace workspace(space.nx(), space.nx(), *prob);
  Solver_t::Results results(space.nx(), *prob);

  Solver_t solver(space, prob);
  solver.setPenalty(0.01);
  solver.use_gauss_newton = true;

  solver.solve(workspace, results, p1, workspace.lamsPrev);
  fmt::print("Results {}\n", results);

  return 0;
}
