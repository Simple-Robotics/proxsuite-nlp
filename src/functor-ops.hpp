#pragma once

#include "lienlp/functor-base.hpp"

#include <utility>


namespace lienlp
{
  /** @brief Compose functors @p F1 and @p F2.
   */
  template<typename F1, typename F2>
  struct ComposeFunctor : DifferentiableFunctor<typename F1::Scalar>
  {
  private:
    const F1& left;
    const F2& right;
  public:
    using Scalar = typename F1::Scalar;
    using Base = DifferentiableFunctor<Scalar>;
    using Base::computeJacobian;
    using Base::vectorHessianProduct;

    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    ComposeFunctor(const F1& left, const F2& right) :
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

  };
  
} // namespace lienlp

