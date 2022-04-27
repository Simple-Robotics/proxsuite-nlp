/** Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#include "proxnlp/modelling/costs/squared-distance.hpp"
#include "proxnlp/modelling/residuals/linear.hpp"
#include "proxnlp/modelling/constraints/equality-constraint.hpp"
#include "proxnlp/modelling/spaces/vector-space.hpp"
#include "proxnlp/solver-base.hpp"

#include "example-base.hpp"

#include <Eigen/QR>


/**
 * Sample a random orthonormal matrix.
 */
template<typename Scalar>
Eigen::Matrix<Scalar, -1, -1> randomOrthogonal(int M, int N)
{
  using MatrixXs = Eigen::Matrix<Scalar, -1, -1>;
  MatrixXs out = MatrixXs::Random(N, N);
  Eigen::FullPivHouseholderQR<Eigen::Ref<MatrixXs>> qr(out);
  Eigen::Matrix<Scalar, -1, -1> Q(qr.matrixQ());
  return Q.template topLeftCorner(M, N);
}


using namespace proxnlp;
using Problem = ProblemTpl<double>;
using EqualityType = EqualityConstraint<double>;
using Constraint = ConstraintObject<double>;

template<int N, int M = 1>
int submain()
{
  using Manifold = VectorSpaceTpl<double>;
  Manifold space(N);
  typename Manifold::PointType p1 = space.rand();

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

  LinearFunction<double> res1(A, b);

  QuadraticDistanceCost<double> cost(space, space.neutral(), Q_);

  std::vector<Problem::ConstraintPtr> cstrs_;
  if (M > 0)
  {
    cstrs_.push_back(std::make_shared<Constraint>(res1, std::make_shared<EqualityType>()));
  }

  auto prob = std::make_shared<Problem>(cost, cstrs_);

  using Solver_t = SolverTpl<double>;
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

int main()
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