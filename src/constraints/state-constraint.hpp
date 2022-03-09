#pragma once

#include "lienlp/constraint-base.hpp"

namespace lienlp {
  
  /**
   * Constraint function to be equal to a given element of a manifold.
   */
  template<class _M>
  struct StateResidual : ConstraintFuncTpl<_M>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using M = _M;
    LIENLP_DEFINE_DYNAMIC_TYPES(typename M::Scalar)
    LIENLP_CSTR_TYPES(Scalar, M::Options)

    using ConstraintFuncTpl<M>::operator();
    using ConstraintFuncTpl<M>::jacobian;

    M& m_manifold;
    VectorXs m_target;

    StateResidual(M& manifold, const VectorXs& target)
      : m_manifold(manifold), m_target(target) {}

    C_t operator()(const VectorXs& x) const
    {
      return m_manifold.diff(m_target, x);
    }

    void jacobian(const VectorXs& x, Jacobian_t& Jout) const
    {
      // set size
      Jout.resize(m_manifold.ndx(), m_manifold.ndx());
      Jout.setZero();
      m_manifold.Jdiff(m_target, x, Jout, 1);
    }

  };

} // namespace lienlp

