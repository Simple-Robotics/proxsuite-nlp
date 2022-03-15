/** Copyright (c) 2022 LAAS-CNRS, INRIA
 * 
 */
#pragma once

#include <Eigen/Cholesky>

#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"


namespace lienlp {

  /** Workspace class, which holds the necessary intermediary data
   * for the solver to function.
   */
  template<typename _Scalar>
  struct SWorkspace
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;

    /// KKT iteration matrix.
    MatrixXs kktMatrix;
    /// KKT iteration right-hand side.
    VectorXs kktRhs;
    /// Primal-dual step.
    VectorXs pdStep;

    /// LDLT storage
    Eigen::LDLT<MatrixXs, Eigen::Upper> ldlt_;

    VectorXs x_prev;
    VectorOfVectors lams_prev;

    SWorkspace(const int nx,
               const int ndx,
               const shared_ptr<Prob_t>& prob)
      :
      kktMatrix(ndx + prob->getNcTotal(), ndx + prob->getNcTotal()),
      kktRhs(ndx + prob->getNcTotal()),
      pdStep(ndx + prob->getNcTotal()),
      ldlt_(ndx + prob->getNcTotal()),
      x_prev(nx)
    {
      kktMatrix.setZero();
      kktRhs.setZero();
      pdStep.setZero();
      x_prev.setZero();

      Prob_t::allocateMultipliers(*prob, lams_prev);
    }
      
  };


} // namespace lienlp

