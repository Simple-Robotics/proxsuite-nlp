/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/function-base.hpp"

#include <utility>

namespace proxnlp {

/** @brief Composition of two functions \f$f \circ g\f$.
 */
template <typename _Scalar> struct ComposeFunctionTpl : C2FunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  using Base = C2FunctionTpl<Scalar>;
  using Base::computeJacobian;
  using Base::vectorHessianProduct;

  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  ComposeFunctionTpl(const shared_ptr<Base> &left,
                     const shared_ptr<Base> &right)
      : Base(right->nx(), right->ndx(), left->nr()), left_(left),
        right_(right) {
    assert(left->nx() == right->nr());
  }

  VectorXs operator()(const ConstVectorRef &x) const {
    return left()(right()(x));
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const {
    left().computeJacobian(right()(x), Jout);
    Jout = Jout * right().computeJacobian(x);
  }
  const Base &left() const { return *left_; }
  const Base &right() const { return *right_; }

private:
  shared_ptr<Base> left_;
  shared_ptr<Base> right_;
};

/// @brief    Compose two function objects.
///
/// @return   ComposeFunctionTpl object representing the composition of @p left
/// and @p right.
template <typename Scalar>
ComposeFunctionTpl<Scalar>
compose(const shared_ptr<C2FunctionTpl<Scalar>> &left,
        const shared_ptr<C2FunctionTpl<Scalar>> &right) {
  return ComposeFunctionTpl<Scalar>(left, right);
}

} // namespace proxnlp
