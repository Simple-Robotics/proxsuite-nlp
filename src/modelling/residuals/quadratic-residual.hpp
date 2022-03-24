#pragma once


#include "lienlp/manifold-base.hpp"
#include "lienlp/residual-base.hpp"


namespace lienlp {
  template<typename M>
  struct QuadraticResidualFunctor : ResidualBase<typename M::Scalar>
  {
    using Scalar = typename M::Scalar;
    LIENLP_RESIDUAL_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Base = ResidualBase<Scalar>;
    using Base::computeJacobian;
    using Base::m_ndx;
    using Base::m_nr;

    const M& m_manifold;
    const MatrixXs m_weights;
    const Scalar m_level;
    const Scalar m_level_squared = m_level * m_level;
    const VectorXs m_target;

    QuadraticResidualFunctor(const M& manifold,
                             const MatrixXs& Q,
                             const Scalar level,
                             const VectorXs& target)
      : Base(manifold.nx(), manifold.ndx(), 1),
        m_manifold(manifold),
        m_weights(Q),
        m_level(level),
        m_target(target)
        {}

    QuadraticResidualFunctor(const M& manifold,
                             const Scalar level,
                             const VectorXs& target)
      : QuadraticResidualFunctor(manifold,
                                 MatrixXs::Identity(manifold.ndx(), manifold.ndx()),
                                 level, target)
        {}

    ReturnType operator()(const ConstVectorRef& x) const
    {
      VectorXs err = m_manifold.difference(m_target, x);
      ReturnType ret(1, 1);
      ret(0, 0) = err.dot(m_weights * err) - m_level_squared;
      return ret;
    }

    void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const
    {
      VectorXs err = m_manifold.difference(m_target, x);
      MatrixXs Jerr(m_manifold.ndx(), m_manifold.ndx());
      m_manifold.Jdifference(m_target, x, Jerr, 1);
      Jout = 2. * (err.transpose() * m_weights) * Jerr;
    }

    void vectorHessianProduct(const ConstVectorRef& x, const ConstVectorRef& v, Eigen::Ref<JacobianType> Hout) const
    {
      MatrixXs Jerr(m_manifold.ndx(), m_manifold.ndx());
      m_manifold.Jdifference(m_target, x, Jerr, 1);
      Hout = 2. * v(0) * (Jerr.transpose() * m_weights * Jerr); // Gauss-Newton approx
    }

  };
} // namespace lienlp

