/** Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#include "lienlp/modelling/costs/squared-distance.hpp"
#include "lienlp/modelling/residuals/linear.hpp"
#include "lienlp/modelling/constraints/equality-constraint.hpp"
#include "lienlp/modelling/spaces/pinocchio-groups.hpp"
#include "lienlp/solver-base.hpp"

#include "example-base.hpp"



using namespace lienlp;
using Prob_t = Problem<double>;
using Equality_t = EqualityConstraint<double>;

template<int N, int M = 1>
int submain()
{
  using Vs = pinocchio::VectorSpaceOperationTpl<N, double>;
  using Man = PinocchioLieGroup<Vs>;
  Man space;
  typename Man::Point_t p0 = space.zero();
  typename Man::Point_t p1 = space.rand();

  Eigen::MatrixXd Qroot(N, 4);
  Qroot.setRandom();
  Eigen::MatrixXd Q_ = Qroot * Qroot.transpose();

  Eigen::MatrixXd A(M, N);
  A.setRandom();
  Eigen::VectorXd b(M);
  b.setRandom();

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
  typename Solver_t::Workspace workspace(space.nx(), space.nx(), *prob);
  typename Solver_t::Results results(space.nx(), *prob);

  Solver_t solver(space, prob);
  solver.setPenalty(1e-3);
  solver.use_gauss_newton = true;

  solver.solve(workspace, results, p1, workspace.lamsPrev);
  fmt::print("Results {}\n\n", results);

  return 0;
}

int main(int argc, const char* argv[])
{
  int s0 = submain<2>();
  int s1 = submain<4>();
  int s2 = submain<4, 3>();
  return 0;
}