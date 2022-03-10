#pragma once

#include "lienlp/cost-function.hpp"
#include "lienlp/constraint-base.hpp"
#include <iostream>

namespace lienlp {
  
/**
 * Cost function which is a weighted squared residual.
 */
template<typename _Scalar>
class QuadResidualCost : public CostFunction<_Scalar>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Scalar = _Scalar;
  using CstrType = ConstraintFuncTpl<Scalar>;  // base constraint func to use
  using M = ManifoldAbstract<Scalar>;
  LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

  M* m_manifold;
  CstrType* m_residual;
  MatrixXs m_weights;

  QuadResidualCost(M* manifold,
                   CstrType* residual,
                   const VectorXs& weightMatrix)
  : m_manifold(manifold), m_residual(residual), m_weights(weightMatrix)
  {}

  template<typename... ResidualArgs>
  QuadResidualCost(const M& manifold,
                   const VectorXs& weightMatrix,
                   ResidualArgs&... args)
                   : QuadResidualCost(manifold, new CstrType(args...),  weightMatrix)
  {}

  Scalar operator()(const VectorXs& x) const
  {
    VectorXs err = m_residual->operator()(x);
    return Scalar(0.5) * err.dot(m_weights * err);
  }

  VectorXs gradient(const VectorXs& x) const
  {
    MatrixXs Jres;
    m_residual->jacobian(x, Jres);
    VectorXs err = m_residual->operator()(x);
    return Jres * (m_weights * err);
  }

  MatrixXs hessian(const VectorXs& x) const
  {
    MatrixXs Jres;
    m_residual->jacobian(x, Jres);
    return Jres.transpose() * (m_weights * Jres);
  }

};

} // namespace lienlp
