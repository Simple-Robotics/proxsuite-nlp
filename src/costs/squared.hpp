#pragma once

#include "lienlp/cost-function.hpp"


namespace lienlp {
  
/**
 * Cost function which is a weighted squared distance from a point x € M,
 * to another target point on the manifold.
 */
template<class M>
class WeightedSquareDistanceCost : public CostFunction<M>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)

  const M& m_manifold;

  using Hess_t = MatrixXs;
  VectorXs m_target;
  MatrixXs m_weights;

  WeightedSquareDistanceCost(const M& manifold,
                             const VectorXs& target,
                             const VectorXs& weightMatrix)
  : m_manifold(manifold), m_target(target), m_weights(weightMatrix) {}

  inline VectorXs residual_(const VectorXs& x) const
  {
    return m_manifold.diff(m_target, x);
  }

  Scalar operator()(const VectorXs& x) const
  {
    VectorXs err = residual_(x);
    return Scalar(0.5) * err.dot(m_weights * err);
  }

  VectorXs gradient(const VectorXs& x) const
  {
    VectorXs err = residual_(x);
    typename M::Jac_t J0, J1;
    m_manifold.Jdiff(m_target, x, J0, J1);
    return J1 * (m_weights * err);
  }

  Hess_t hessian(const VectorXs& x) const
  {
    typename M::Jac_t J0, J1;
    m_manifold.Jdiff(m_target, x, J0, J1);
    return J1.transpose() * (m_weights * J1);
  }

};

} // namespace lienlp
