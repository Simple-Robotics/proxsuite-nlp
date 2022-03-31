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
    VectorOfVectors lamsPrev;
    VectorOfVectors lamsTrial;

    /// Residuals

    VectorXs dualResidual;
    Scalar dualInfeas;

    VectorOfVectors primalResiduals;
    Scalar primalInfeas;

    /// Objective function gradient.
    VectorXs objectiveGradient;
    /// Merit function gradient.
    VectorXs meritGradient;
    /// Objective function Hessian.
    MatrixXs objectiveHessian;

    std::vector<MatrixXs> cstrJacobians;
    std::vector<MatrixXs> cstrVectorHessProd;
    /// First-order multipliers \f$\mathrm{proj}(\lambda_e + c / \mu)\f$
    VectorOfVectors lamsPlus;
    /// Pre-projected multipliers.
    VectorOfVectors lamsPlusPre;
    /// Primal-dual multiplier estimates (from the pdBCL algorithm)
    VectorOfVectors lamsPDAL;
    /// Subproblem proximal dual error.
    VectorOfVectors subproblemDualErr;


    SWorkspace(const int nx,
               const int ndx,
               const Problem& prob)
      :
      kktMatrix(ndx + prob.getTotalConstraintDim(), ndx + prob.getTotalConstraintDim()),
      kktRhs(ndx + prob.getTotalConstraintDim()),
      pdStep(ndx + prob.getTotalConstraintDim()),
      signature(ndx + prob.getTotalConstraintDim()),
      ldlt_(ndx + prob.getTotalConstraintDim()),
      xPrev(nx),
      xTrial(nx),
      dualResidual(ndx),
      objectiveGradient(ndx),
      meritGradient(ndx),
      objectiveHessian(ndx, ndx)
    {
      init(prob);
    }

    void init(const Problem& prob)
    {
      kktMatrix.setZero();
      kktRhs.setZero();
      pdStep.setZero();
      signature.setConstant(1);

      xPrev.setZero();
      xTrial.setZero();
      helpers::allocateMultipliersOrResiduals(prob, lamsPrev);
      helpers::allocateMultipliersOrResiduals(prob, lamsTrial);

      dualResidual.setZero();
      helpers::allocateMultipliersOrResiduals(prob, primalResiduals);  // not multipliers but same dims

      objectiveGradient.setZero();
      meritGradient.setZero();
      objectiveHessian.setZero();

      helpers::allocateMultipliersOrResiduals(prob, lamsPlusPre);
      helpers::allocateMultipliersOrResiduals(prob, lamsPlus);
      helpers::allocateMultipliersOrResiduals(prob, lamsPDAL);
      helpers::allocateMultipliersOrResiduals(prob, subproblemDualErr);


      const std::size_t nc = prob.getNumConstraints();
      const int ndx = prob.m_cost.ndx();

      cstrJacobians.reserve(nc);
      cstrVectorHessProd.reserve(nc);

      for (std::size_t i = 0; i < nc; i++)
      {
        auto cstr = prob.getConstraint(i);
        int nr = cstr->nr();
        cstrJacobians.push_back(MatrixXs::Zero(nr, ndx));
        cstrVectorHessProd.push_back(MatrixXs::Zero(ndx, ndx));
      }

    }
      
  };


} // namespace lienlp

