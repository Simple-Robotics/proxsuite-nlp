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

    /// Newton iteration variables

    /// KKT iteration matrix.
    MatrixXs kktMatrix;
    /// KKT iteration right-hand side.
    VectorXs kktRhs;
    /// Primal-dual step.
    VectorXs pdStep;

    /// LDLT storage
    Eigen::LDLT<MatrixXs, Eigen::Lower> ldlt_;

    //// Proximal parameters

    VectorXs xPrev;
    VectorOfVectors lamsPrev;

    //// Meta

    std::size_t numIters;
    Scalar objective;

    /// Residuals

    VectorXs dualResidual;
    Scalar dualInfeas;

    VectorOfVectors primalResiduals;
    std::vector<Scalar> primalInfeas;

    /// tmp

    VectorXs tmpObjGrad;
    MatrixXs tmpObjHess;

    std::vector<MatrixXs> tmpCstrJacobians;
    std::vector<MatrixXs> tmpCstrVectorHessProd;
    VectorOfVectors tmpLamsPlus;
    VectorOfVectors auxProxDualErr;


    SWorkspace(const int nx,
               const int ndx,
               const Prob_t& prob)
      :
      kktMatrix(ndx + prob.getNcTotal(), ndx + prob.getNcTotal()),
      kktRhs(ndx + prob.getNcTotal()),
      pdStep(ndx + prob.getNcTotal()),
      numIters(0),
      ldlt_(ndx + prob.getNcTotal()),
      xPrev(nx),
      tmpObjGrad(ndx),
      tmpObjHess(ndx, ndx)
    {
      init(prob);
    }

    void init(const Prob_t& prob)
    {
      kktMatrix.setZero();
      kktRhs.setZero();
      pdStep.setZero();
      xPrev.setZero();

      dualResidual.setZero();
      Prob_t::allocateMultipliers(prob, primalResiduals);  // not multipliers but same dims

      tmpObjGrad.setZero();
      tmpObjHess.setZero();

      objective = 0.;

      Prob_t::allocateMultipliers(prob, lamsPrev);
      Prob_t::allocateMultipliers(prob, tmpLamsPlus);
      Prob_t::allocateMultipliers(prob, auxProxDualErr);
    }
      
  };


} // namespace lienlp

