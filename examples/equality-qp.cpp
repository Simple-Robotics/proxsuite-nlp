/** Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#include "lienlp/modelling/costs/squared-distance.hpp"
#include "lienlp/modelling/residuals/linear.hpp"
#include "lienlp/modelling/constraints/equality-constraint.hpp"
#include "lienlp/modelling/spaces/pinocchio-groups.hpp"
#include "lienlp/solver-base.hpp"

#include "example-base.hpp"

#include <Eigen/QR>


/**
 * Sample a random orthonormal matrix.
 */
template<typename Scalar>
Eigen::Matrix<Scalar, -1, -1> randomOrthogonal(int M, int N)
{
  using ReturnType = Eigen::Matrix<Scalar, -1, -1>;
  ReturnType out = ReturnType::Random(N, N);
  Eigen::FullPivHouseholderQR<Eigen::Ref<ReturnType>> qr(out);
  Eigen::Matrix<Scalar, -1, -1> Q(qr.matrixQ());
  return Q.template topLeftCorner(M, N);
}


using namespace lienlp;
using Prob_t = Problem<double>;
using EqualityType = EqualityConstraint<double>;

template<int N, int M = 1>
int submain()
{
  using Vs = pinocchio::VectorSpaceOperationTpl<N, double>;
  using Man = PinocchioLieGroup<Vs>;
  Man space;
  typename Man::Point_t p1 = space.rand();

  Eigen::MatrixXd Qroot(N, N + 1);
  Qroot.setRandom();
  Eigen::MatrixXd Q_ = Qroot * Qroot.transpose() / N;

  Eigen::MatrixXd A(M, N);
  A.setZero();
  if (M > 0)
  {
    A = randomOrthogonal<double>(M, N);
  }
  Eigen::VectorXd b(M);
  b.setRandom();

  LinearResidual<double> res1(A, b);

  QuadDistanceCost<Man> cost(space, Q_);

  auto cstr1 = std::make_shared<EqualityType>(res1);
  std::vector<Prob_t::CstrPtr> cstrs_;
  if (M > 0) cstrs_.push_back(cstr1);

  auto prob = std::make_shared<Prob_t>(cost, cstrs_);

  using Solver_t = Solver<Man>;
  typename Solver_t::Workspace workspace(space.nx(), space.nx(), *prob);
  typename Solver_t::Results results(space.nx(), *prob);

  Solver_t solver(space, prob);
  solver.setPenalty(1e-4);
  solver.rho = 1e-8;
  solver.use_gauss_newton = true;

  solver.solve(workspace, results, p1, workspace.lamsPrev);
  fmt::print("Results {}\n\n", results);

  return 0;
}

int main(int argc, const char* argv[])
{
  submain<2>();
  submain<4>();
  submain<4, 3>();
  submain<10, 4>();
  submain<10, 6>();
  submain<20, 1>();
  submain<20, 4>();
  submain<50, 0>();
  submain<50, 10>();
  submain<100, 50>();
  submain<200, 42>();
  return 0;
}