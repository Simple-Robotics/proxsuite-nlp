#pragma once

#include "lienlp/cost-function.hpp"
#include "lienlp/functor-base.hpp"


namespace lienlp
{
  /** @brief  Convert a cost function to a functor.
   *  @todo    Deprecate this class after making cost functions functors themselves.
   */
  template<typename C>
  struct convert_cost_to_functor
  {
    using Scalar = typename C::Scalar;
    using in_type = C;
    struct out_type : DifferentiableFunctor<Scalar>
    {
      LIENLP_FUNCTOR_TYPEDEFS(Scalar)

      using Base = DifferentiableFunctor<Scalar>;
      in_type base_cost;

      out_type(const C& in) : Base(in.nx(), in.ndx(), 1), base_cost(in) {}

      ReturnType operator()(const ConstVectorRef& x) const
      {
        ReturnType out(1, 1);
        out << base_cost(x);
        return out;
      }

      void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const
      {
        Jout = base_cost.computeGradient(x).transpose();
      }

      void vectorHessianProduct(const ConstVectorRef& x, const ConstVectorRef&, Eigen::Ref<JacobianType> Hout) const
      {
        base_cost.computeHessian(x, Hout);
      }

    };

  };
  
} // namespace lienlp

