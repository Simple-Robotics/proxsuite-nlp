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
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;

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
    VectorOfVectors auxProxDualErr;


    SWorkspace(const int nx,
               const int ndx,
               const Prob_t& prob)
      :
      kktMatrix(ndx + prob.getNcTotal(), ndx + prob.getNcTotal()),
      kktRhs(ndx + prob.getNcTotal()),
      pdStep(ndx + prob.getNcTotal()),
      signature(ndx + prob.getNcTotal()),
      ldlt_(ndx + prob.getNcTotal()),
      xPrev(nx),
      xTrial(nx),
      dualResidual(ndx),
      objectiveGradient(ndx),
      meritGradient(ndx),
      objectiveHessian(ndx, ndx)
    {
      init(prob);
    }

    void init(const Prob_t& prob)
    {
      kktMatrix.setZero();
      kktRhs.setZero();
      pdStep.setZero();
      signature.setConstant(1);

      xPrev.setZero();
      xTrial.setZero();
      Prob_t::allocateMultipliers(prob, lamsPrev);
      Prob_t::allocateMultipliers(prob, lamsTrial);

      dualResidual.setZero();
      Prob_t::allocateMultipliers(prob, primalResiduals);  // not multipliers but same dims

      objectiveGradient.setZero();
      meritGradient.setZero();
      objectiveHessian.setZero();

      Prob_t::allocateMultipliers(prob, lamsPlusPre);
      Prob_t::allocateMultipliers(prob, lamsPlus);
      Prob_t::allocateMultipliers(prob, lamsPDAL);
      Prob_t::allocateMultipliers(prob, auxProxDualErr);


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

