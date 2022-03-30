#pragma once

#include "lienlp/functor-base.hpp"

#include <utility>


namespace lienlp
{

  /** @brief Compose two functors.
   */
  template<typename _Scalar>
  struct ComposeFunctor : DifferentiableFunctor<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    using Base = DifferentiableFunctor<Scalar>;
    using Base::computeJacobian;
    using Base::vectorHessianProduct;

    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    ComposeFunctor(const Base& left, const Base& right) :
        DifferentiableFunctor<Scalar>(right.nx(), right.ndx(), left.nr())
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

