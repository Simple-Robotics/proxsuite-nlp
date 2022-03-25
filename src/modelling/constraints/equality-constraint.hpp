#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/constraint-base.hpp"


namespace lienlp
{
  
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
    using Scalar = _Scalar;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    using Base = ConstraintSetBase<Scalar>;
    using Base::operator();
    using Active_t = typename Base::Active_t;
    using functor_t = typename Base::functor_t;

    EqualityConstraint(const functor_t& func)
      : Base(func) {}

    inline ReturnType projection(const ConstVectorRef& z) const
    {
      return z * Scalar(0.);
    }

    inline void applyProjectionJacobian(const ConstVectorRef&, MatrixRef Jout) const
    {
      Jout.setZero();
    }

    inline void applyNormalConeProjectionJacobian(const ConstVectorRef&, MatrixRef) const
    {
      return;  // do nothing
    }

    inline void computeActiveSet(const ConstVectorRef&, Eigen::Ref<Active_t> out) const
    {
      out.array() = true;
    }
  };

} // namespace lienlp

