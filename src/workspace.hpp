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
    using VecBool = Eigen::Matrix<bool, -1, 1>;

    /// Newton iteration variables

    /// KKT iteration matrix.
    MatrixXs kktMatrix;
    /// KKT iteration right-hand side.
    VectorXs kktRhs;
    /// Primal-dual step.
    VectorXs pdStep;
    /// Signature of the matrix
    VecBool signature;

    /// LDLT storage
    Eigen::LDLT<MatrixXs, Eigen::Lower> ldlt_;

    //// Proximal parameters

    VectorXs xPrev;
    VectorOfVectors lamsPrev;

    /// Residuals

    VectorXs dualResidual;
    Scalar dualInfeas;

    VectorOfVectors primalResiduals;
    Scalar primalInfeas;

    /// tmp

    VectorXs objectiveGradient;
    MatrixXs objectiveHessian;

    std::vector<MatrixXs> cstrJacobians;
    std::vector<MatrixXs> cstrVectorHessProd;
    /// cached 1st-order multipliers \f$\mathrm{proj}(\lambda_e + c / mu)\f$
    VectorOfVectors lamsPlus;
    /// cached PDAL estimates
    VectorOfVectors lamsPDAL;
    /// dual prox error \f$\mu (\lambda^+ - \lambda)\f$
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
      objectiveGradient(ndx),
      objectiveHessian(ndx, ndx)
    {
      init(prob);
    }

    void init(const Prob_t& prob)
    {
      kktMatrix.setZero();
      kktRhs.setZero();
      pdStep.setZero();
      xPrev.setZero();
      signature.setConstant(false);

      dualResidual.setZero();

      objectiveGradient.setZero();
      objectiveHessian.setZero();

      Prob_t::allocateMultipliers(prob, primalResiduals);  // not multipliers but same dims
      Prob_t::allocateMultipliers(prob, lamsPrev);
      Prob_t::allocateMultipliers(prob, lamsPlus);
      Prob_t::allocateMultipliers(prob, lamsPDAL);
      Prob_t::allocateMultipliers(prob, auxProxDualErr);


      const std::size_t nc = prob.getNumConstraints();
      const int ndx = prob.m_cost.ndx();

      cstrJacobians.reserve(nc);
      cstrVectorHessProd.reserve(nc);

      for (std::size_t i = 0; i < nc; i++)
      {
        auto cstr = prob.getCstr(i);
        int nr = cstr->nr();
        cstrJacobians.push_back(MatrixXs::Zero(nr, ndx));
        cstrVectorHessProd.push_back(MatrixXs::Zero(ndx, ndx));
      }

    }
      
  };


} // namespace lienlp

