#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/constraint-base.hpp"


namespace lienlp {
  
  template<class _Scalar>
  struct EqualityConstraint : ConstraintFormatBaseTpl<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_CSTR_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using ConstraintFormatBaseTpl<Scalar>::operator();
    using functor_t = ConstraintFuncTpl<Scalar>;

    EqualityConstraint<Scalar>(const functor_t& func, const int& nc)
      : ConstraintFormatBaseTpl<Scalar>(func, nc) {}

    C_t projection(const VectorXs& x) const
    {
      return VectorXs::Zero(x.size());
    }

    Jacobian_t Jprojection(const VectorXs& x) const
    {
      return MatrixXs::Zero(x.size(), x.size());
    }

  };

} // namespace lienlp

