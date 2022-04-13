/** Copyright (c) 2022 LAAS-CNRS, INRIA
 * 
 */
#pragma once

#include <Eigen/Cholesky>

#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"


namespace lienlp
{

  /** Workspace class, which holds the necessary intermediary data
   * for the solver to function.
   */
  template<typename _Scalar>
  struct SWorkspace
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using Problem = ProblemTpl<Scalar>;

    /// Newton iteration variables

    const int ndx;
    const int numcstr;

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

    //// Proximal parameters

    VectorXs xPrev;
    VectorXs xTrial;
    VectorXs lamsPrev_data;
    VectorXs lamsTrial_data;
    VectorOfRef lamsPrev;
    VectorOfRef lamsTrial;

    /// Residuals

    VectorXs dualResidual;
    VectorXs primalResiduals_data;
    VectorOfRef primalResiduals;

    /// Objective function gradient.
    VectorXs objectiveGradient;
    /// Objective function Hessian.
    MatrixXs objectiveHessian;
    /// Merit function gradient.
    VectorXs meritGradient;

    MatrixXs jacobians_data;
    MatrixXs hessians_data;
    std::vector<MatrixRef> cstrJacobians;
    std::vector<MatrixRef> cstrVectorHessProd;

    VectorXs lamsPlusPre_d;
    VectorXs lamsPlus_d;
    VectorXs lamsPDAL_d;
    VectorXs subproblemDualErr_d;

    /// First-order multipliers \f$\mathrm{proj}(\lambda_e + c / \mu)\f$
    VectorOfRef lamsPlus;
    /// Pre-projected multipliers.
    VectorOfRef lamsPlusPre;
    /// Primal-dual multiplier estimates (from the pdBCL algorithm)
    VectorOfRef lamsPDAL;
    /// Subproblem proximal dual error.
    VectorOfRef subproblemDualErr;


    SWorkspace(const int nx,
               const int ndx,
               const Problem& prob)
      : ndx(ndx)
      , numcstr(prob.getTotalConstraintDim())
      , kktMatrix(ndx + numcstr, ndx + numcstr)
      , kktRhs(ndx + numcstr)
      , pdStep(ndx + numcstr)
      , signature(ndx + numcstr)
      , ldlt_(ndx + numcstr)
      , xPrev(nx)
      , xTrial(nx)
      , lamsPrev_data(numcstr)
      , lamsTrial_data(numcstr)
      , dualResidual(ndx)
      , primalResiduals_data(numcstr)
      , objectiveGradient(ndx)
      , objectiveHessian(ndx, ndx)
      , meritGradient(ndx)
      , jacobians_data(numcstr, ndx)
      , hessians_data(numcstr * ndx, ndx)
      , lamsPlus_d(numcstr)
      , lamsPDAL_d(numcstr)
      , subproblemDualErr_d(numcstr)
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

      dualResidual.setZero();
      helpers::allocateMultipliersOrResiduals(prob, primalResiduals_data, primalResiduals);  // not multipliers but same dims

      objectiveGradient.setZero();
      objectiveHessian.setZero();
      meritGradient.setZero();
      jacobians_data.setZero();
      hessians_data.setZero();

      helpers::allocateMultipliersOrResiduals(prob, lamsPlusPre_d, lamsPlusPre);
      helpers::allocateMultipliersOrResiduals(prob, lamsPlus_d, lamsPlus);
      helpers::allocateMultipliersOrResiduals(prob, lamsPDAL_d, lamsPDAL);
      helpers::allocateMultipliersOrResiduals(prob, subproblemDualErr_d, subproblemDualErr);


      const std::size_t nc = prob.getNumConstraints();
      const int ndx = prob.m_cost.ndx();

      cstrJacobians.reserve(nc);
      cstrVectorHessProd.reserve(nc);

      int cursor = 0;
      int nr = 0;
      for (std::size_t i = 0; i < nc; i++)
      {
        auto cstr = prob.getConstraint(i);
        nr = cstr->nr();
        cstrJacobians.push_back(jacobians_data.middleRows(cursor, nr));
        cstrVectorHessProd.push_back(hessians_data.middleRows(cursor * ndx, ndx));
        cursor += nr;
      }

    }
      
  };


} // namespace lienlp

