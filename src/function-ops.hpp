#pragma once

#include "lienlp/function-base.hpp"

#include <utility>


namespace lienlp
{

  /** @brief Compose two functions.
   */
  template<typename _Scalar>
  struct ComposeFunction : C2Function<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    using Base = C2Function<Scalar>;
    using Base::computeJacobian;
    using Base::vectorHessianProduct;

    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    ComposeFunction(const Base& left, const Base& right) :
        C2Function<Scalar>(right.nx(), right.ndx(), left.nr())
      , left(left), right(right) {}

    ReturnType operator()(const ConstVectorRef& x) const
    {
      return left(right(x));
    }

    void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const
    {
      left.computeJacobian(right(x), Jout);
      Jout = Jout * right.computeJacobian(x);
    }
  private:
    const Base& left;
    const Base& right;

  };
  
} // namespace lienlp

