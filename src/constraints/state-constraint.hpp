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

    using ConstraintFuncTpl<Scalar>::operator();
    using ConstraintFuncTpl<Scalar>::jacobian;

    M* m_manifold;

    VectorXs m_target;

    StateResidual(M* manifold, const VectorXs& target)
      : m_manifold(manifold), m_target(target) {}

    C_t operator()(const VectorXs& x) const
    {
      return m_manifold->difference(m_target, x);
    }

    void jacobian(const VectorXs& x, Jacobian_t& Jout) const
    {
      // set size
      Jout.resize(m_manifold->ndx(), m_manifold->ndx());
      Jout.setZero();
      m_manifold->Jdifference(m_target, x, Jout, 1);
    }

  };

} // namespace lienlp

