#pragma once

#include "proxnlp/function-base.hpp"

#include <utility>


namespace proxnlp
{

  /** @brief Composition of two functions \f$f \circ g\f$.
   */
  template<typename _Scalar>
  struct ComposeFunctionTpl : C2FunctionTpl<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    using Base = C2FunctionTpl<Scalar>;
    using Base::computeJacobian;
    using Base::vectorHessianProduct;

    PROXNLP_FUNCTION_TYPEDEFS(Scalar);

    ComposeFunctionTpl(const Base& left, const Base& right)
      : Base(right.nx(), right.ndx(), left.nr())
      , left(left), right(right) {}

    ReturnType operator()(const ConstVectorRef& x) const
    {
      return left(right(x));
    }

    void computeJacobian(const ConstVectorRef& x, MatrixRef Jout) const
    {
      left.computeJacobian(right(x), Jout);
      Jout = Jout * right.computeJacobian(x);
    }
  private:
    const Base& left;
    const Base& right;

  };


  /// @brief    Compose two function objects.
  ///
  /// @return   ComposeFunctionTpl object representing the composition of @p left and @p right.
  template<typename Scalar>
  ComposeFunctionTpl<Scalar> compose(const C2FunctionTpl<Scalar>& left, const C2FunctionTpl<Scalar>& right)
  {
    return ComposeFunctionTpl<Scalar>(left, right);
  }

  
} // namespace proxnlp

