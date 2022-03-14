#pragma once

#include "lienlp/constraint-base.hpp"

namespace lienlp {
  
  /**
   * Constraint function to be equal to a given element of a manifold.
   * This is templated on the manifold.
   */
  template<typename M>
  struct StateResidual : ConstraintFuncTpl<typename M::Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = typename M::Scalar;
    LIENLP_CSTR_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    using Parent = ConstraintFuncTpl<Scalar>;
    using Parent::operator();
    using Parent::jacobian;
    using Parent::m_ndx;
    using Parent::m_nc;

    M* m_manifold;

    VectorXs m_target;

    StateResidual(M* manifold, const ConstVectorRef& target)
      : m_manifold(manifold), m_target(target), Parent(manifold->ndx(), manifold->ndx()) {}

    C_t operator()(const ConstVectorRef& x) const
    {
      return m_manifold->difference(m_target, x);
    }

    void jacobian(const ConstVectorRef& x, Jacobian_t& Jout) const
    {
      m_manifold->Jdifference(m_target, x, Jout, 1);
    }

  };

} // namespace lienlp

