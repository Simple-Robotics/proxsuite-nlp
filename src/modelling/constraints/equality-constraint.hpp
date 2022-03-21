#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/constraint-base.hpp"


namespace lienlp {
  
  /**
   * @brief   Equality constraints \f$c(x) = 0\f$.
   * 
   * @details This class implements the set associated with equality constraints\f$ c(x) = 0 \f$,
   *          where \f$c : \calX \to \RR^p\f$ is a residual function.
   */
  template<typename _Scalar>
  struct EqualityConstraint : ConstraintSetBase<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_RESIDUAL_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Base = ConstraintSetBase<Scalar>;
    using Base::operator();
    using Active_t = typename Base::Active_t;
    using functor_t = typename Base::functor_t;

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
  private:
    ReturnType proj_;
    JacobianType Jproj_;
  };

} // namespace lienlp

