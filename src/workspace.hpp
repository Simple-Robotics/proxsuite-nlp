/* Copyright (C) 2022 LAAS-CNRS, INRIA
 * 
 */
#pragma once

#include "proxnlp/problem-base.hpp"

#include <Eigen/Cholesky>

namespace proxnlp
{

  /** Workspace class, which holds the necessary intermediary data
   * for the solver to function.
   */
  template<typename _Scalar>
  struct WorkspaceTpl
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
    using Problem = ProblemTpl<Scalar>;

    /// Newton iteration variables

    const int ndx;
    const std::size_t numblocks;    // number of constraint blocks
    const int numdual;              // total constraint dim

    /// KKT iteration matrix.
    MatrixXs kktMatrix;
    /// KKT iteration right-hand side.
    VectorXs kktRhs;
    /// Primal-dual step.
    VectorXs pdStep;
    /// Signature of the matrix
    Eigen::VectorXi signature;

    /// LDLT storage
    Eigen::LDLT<MatrixXs, Eigen::Lower> ldlt_;

    //// Data for proximal algorithm

    VectorXs xPrev;
    VectorXs xTrial;
    VectorXs lamsPrev_data;
    VectorXs lamsTrial_data;
    VectorOfRef lamsPrev;
    VectorOfRef lamsTrial;

    VectorXs prox_grad;
    MatrixXs prox_hess;

    /// Residuals

    /// Dual residual: gradient of the Lagrangian function
    VectorXs dualResidual;
    VectorXs cstr_values_data;
    /// Values of each constraint
    VectorOfRef cstrValues;

    /// Objective function gradient.
    VectorXs objectiveGradient;
    /// Objective function Hessian.
    MatrixXs objectiveHessian;
    /// Merit function gradient.
    VectorXs meritGradient;

    MatrixXs jacobians_data;
    MatrixXs hessians_data;
    std::vector<MatrixRef> cstrJacobians;
    std::vector<MatrixRef> cstrVectorHessianProd;

    VectorXs lamsPlusPre_data;
    VectorXs lamsPlus_data;
    VectorXs lamsPDAL_data;
    VectorXs subproblemDualErr_data;

    /// First-order multipliers \f$\mathrm{proj}(\lambda_e + c / \mu)\f$
    VectorOfRef lamsPlus;
    /// Pre-projected multipliers.
    VectorOfRef lamsPlusPre;
    /// Primal-dual multiplier estimates (from the pdBCL algorithm)
    VectorOfRef lamsPDAL;
    /// Subproblem proximal dual error.
    VectorOfRef subproblemDualErr;

    std::vector<Scalar> ls_alphas;
    std::vector<Scalar> ls_values;
    Scalar d1;

    WorkspaceTpl(const int nx,
               const int ndx,
               const Problem& prob)
      : ndx(ndx)
      , numblocks(prob.getNumConstraints())
      , numdual(prob.getTotalConstraintDim())
      , kktMatrix(ndx + numdual, ndx + numdual)
      , kktRhs(ndx + numdual)
      , pdStep(ndx + numdual)
      , signature(ndx + numdual)
      , ldlt_(kktMatrix)
      , xPrev(nx)
      , xTrial(nx)
      , lamsPrev_data(numdual)
      , lamsTrial_data(numdual)
      , prox_grad(ndx)
      , prox_hess(ndx, ndx)
      , dualResidual(ndx)
      , cstr_values_data(numdual)
      , objectiveGradient(ndx)
      , objectiveHessian(ndx, ndx)
      , meritGradient(ndx)
      , jacobians_data(numdual, ndx)
      , hessians_data((int)numblocks * ndx, ndx)
      , lamsPlus_data(numdual)
      , lamsPDAL_data(numdual)
      , subproblemDualErr_data(numdual)
    {
      init(prob);
    }

    void init(const Problem& prob)
    {
      kktMatrix.setZero();
      kktRhs.setZero();
      pdStep.setZero();
      signature.setZero();

      xPrev.setZero();
      xTrial.setZero();
      helpers::allocateMultipliersOrResiduals(prob, lamsPrev_data, lamsPrev);
      helpers::allocateMultipliersOrResiduals(prob, lamsTrial_data, lamsTrial);
      prox_grad.setZero();
      prox_hess.setZero();

      dualResidual.setZero();
      helpers::allocateMultipliersOrResiduals(prob, cstr_values_data, cstrValues);  // not multipliers but same dims

      objectiveGradient.setZero();
      objectiveHessian.setZero();
      meritGradient.setZero();
      jacobians_data.setZero();
      hessians_data.setZero();

      helpers::allocateMultipliersOrResiduals(prob, lamsPlusPre_data, lamsPlusPre);
      helpers::allocateMultipliersOrResiduals(prob, lamsPlus_data, lamsPlus);
      helpers::allocateMultipliersOrResiduals(prob, lamsPDAL_data, lamsPDAL);
      helpers::allocateMultipliersOrResiduals(prob, subproblemDualErr_data, subproblemDualErr);

      cstrJacobians.reserve(numblocks);
      cstrVectorHessianProd.reserve(numblocks);

      int cursor = 0;
      int nr = 0;
      for (std::size_t i = 0; i < numblocks; i++)
      {
        cursor = prob.getIndex(i);
        nr = prob.getConstraintDim(i);
        cstrJacobians.emplace_back(jacobians_data.middleRows(cursor, nr));
        cstrVectorHessianProd.emplace_back(hessians_data.middleRows((int)i * ndx, ndx));
      }

    }

  };


} // namespace proxnlp

