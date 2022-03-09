#pragma once

#include "lienlp/cost-function.hpp"
#include "lienlp/constraint-base.hpp"
#include <iostream>

namespace lienlp {
  
/**
 * Cost function which is a weighted squared residual.
 */
template<class CstrType>
class QuadResidualCost : public CostFunction<typename CstrType::M>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using M = typename CstrType::M;
  LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)

  const M& m_manifold;
  const CstrType& m_residual;
  MatrixXs m_weights;

  QuadResidualCost(const M& manifold,
                   const CstrType& residual,
                   const VectorXs& weightMatrix)
  : m_manifold(manifold), m_residual(residual), m_weights(weightMatrix)
  {}

  template<typename... ResidualArgs>
  QuadResidualCost(const M& manifold,
                   const VectorXs& weightMatrix,
                   ResidualArgs&... args)
                   : QuadResidualCost(manifold, CstrType(args...),  weightMatrix)
  {}

  Scalar operator()(const VectorXs& x) const
  {
    VectorXs err = m_residual(x);
    return Scalar(0.5) * err.dot(m_weights * err);
  }

  VectorXs gradient(const VectorXs& x) const
  {
    MatrixXs Jres;
    m_residual.jacobian(x, Jres);
    VectorXs err = m_residual(x);
    return Jres * (m_weights * err);
  }

  MatrixXs hessian(const VectorXs& x) const
  {
    MatrixXs Jres;
    m_residual.jacobian(x, Jres);
    return Jres.transpose() * (m_weights * Jres);
  }

};

} // namespace lienlp
