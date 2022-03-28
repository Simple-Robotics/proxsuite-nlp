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
    using S1 = typename F1::Scalar;
    using S2 = typename F2::Scalar;
    using Scalar = decltype(std::declval<S1>() * std::declval<S2>());

    LIENLP_FUNCTOR_TYPEDEFS(Scalar)
    using Base = DifferentiableFunctor<Scalar>;


    ComposeFunctor(const F1& left, const F2& right) : left(left), right(right) {}

    ReturnType operator()(const ConstVectorRef& x) const
    {
      return left(right(x));
    }

    void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const
    {
      left.computeJacobian(right(x), Jout);
      Jout = Jout * right.computeJacobian(x);
    }

  }
  
} // namespace lienlp

