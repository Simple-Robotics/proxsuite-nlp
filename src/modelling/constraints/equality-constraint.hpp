#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/constraint-base.hpp"


namespace lienlp {
  
  template<typename _Scalar>
  struct EqualityConstraint : ConstraintSetBase<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_RESIDUAL_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Base = ConstraintSetBase<Scalar>;
    using Base::operator();
    using Base::computeJacobian;
    using Active_t = typename Base::Active_t;
    using functor_t = typename Base::functor_t;

    // TODO MAKE CONST
    ReturnType proj_;
    JacobianType Jproj_;

    EqualityConstraint(const functor_t& func)
      : ConstraintSetBase<Scalar>(func),
        proj_(func.nr()),
        Jproj_(func.nr(), func.nr())
      {
        proj_.setZero();
        Jproj_.setZero();
      }

    ReturnType projection(const ConstVectorRef& z) const
    {
      return proj_;
    }

    JacobianType Jprojection(const ConstVectorRef& z) const
    {
      return Jproj_;
    }

    void computeActiveSet(const ConstVectorRef& z, Active_t& out) const
    {
      out.array() = true;
    }
  };

} // namespace lienlp

