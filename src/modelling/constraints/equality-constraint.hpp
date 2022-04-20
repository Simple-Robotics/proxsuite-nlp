#pragma once

#include "proxnlp/constraint-base.hpp"


namespace proxnlp
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
    PROXNLP_FUNCTOR_TYPEDEFS(Scalar)

    using Base = ConstraintSetBase<Scalar>;
    using ActiveType = typename Base::ActiveType;
    using FunctionType = typename Base::FunctionType;

    bool disableGaussNewton() const { return true; }

    explicit EqualityConstraint(const FunctionType& func)
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

    inline void computeActiveSet(const ConstVectorRef&, Eigen::Ref<ActiveType> out) const
    {
      out.array() = true;
    }
  };

} // namespace proxnlp

