#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/constraint-base.hpp"


namespace lienlp {
  
  template<typename _Scalar>
  struct EqualityConstraint : ConstraintFormatBaseTpl<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_CSTR_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Parent = ConstraintFormatBaseTpl<Scalar>;
    using Parent::operator();
    using Parent::jacobian;
    using functor_t = typename Parent::functor_t;

    VectorXs proj_;
    VectorXs Jproj_;

    EqualityConstraint(const functor_t& func)
      : ConstraintFormatBaseTpl<Scalar>(func) {
        proj_ = VectorXs::Zero(func.getDim());
        Jproj_ = VectorXs::Zero(func.getDim(), func.ndx());
      }

    C_t projection(const ConstVectorRef& z) const
    {
      return proj_;
    }

    Jacobian_t Jprojection(const ConstVectorRef& z) const
    {
      return Jproj_;
    }

    void computeActiveSet(const ConstVectorRef& z, Active_t& out) const
    {
      out.array() = false;
    }
  };

} // namespace lienlp

