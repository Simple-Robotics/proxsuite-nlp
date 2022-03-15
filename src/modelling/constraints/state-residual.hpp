#pragma once

#include "lienlp/residual-base.hpp"


namespace lienlp {
  
  /**
   * Constraint function to be equal to a given element of a manifold.
   * This is templated on the manifold.
   */
  template<typename M>
  struct StateResidual : ResidualBase<typename M::Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = typename M::Scalar;
    LIENLP_RESIDUAL_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    using Base = ResidualBase<Scalar>;
    using Base::operator();
    using Base::computeJacobian;
    using Base::m_ndx;
    using Base::m_nr;

    M* m_manifold;

    VectorXs m_target;

    StateResidual(M* manifold, const ConstVectorRef& target)
      : Base(manifold->nx(), manifold->ndx(), manifold->ndx()),
        m_manifold(manifold), m_target(target)
      {}

    ReturnType operator()(const ConstVectorRef& x) const
    {
      return m_manifold->difference(m_target, x);
    }

    void computeJacobian(const ConstVectorRef& x, JacobianType& Jout) const
    {
      m_manifold->Jdifference(m_target, x, Jout, 1);
    }

  };

} // namespace lienlp

