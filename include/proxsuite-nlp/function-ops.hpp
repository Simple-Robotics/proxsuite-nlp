/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/function-base.hpp"

#include <utility>

namespace proxsuite {
namespace nlp {

/** @brief Composition of two functions \f$f \circ g\f$.
 */
template <typename _Scalar> struct ComposeFunctionTpl : C2FunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  using Base = C2FunctionTpl<Scalar>;
  using Base::computeJacobian;
  using Base::vectorHessianProduct;

  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  ComposeFunctionTpl(const shared_ptr<Base> &left,
                     const shared_ptr<Base> &right)
      : Base(right->nx(), right->ndx(), left->nr()), left_(left),
        right_(right) {
    if (left->nx() != right->nr()) {
      PROXSUITE_NLP_RUNTIME_ERROR(fmt::format(
          "Incompatible dimensions ({:d} and {:d}).", left->nx(), right->nr()));
    }
    assert(left->nx() == right->nr());
  }

  VectorXs operator()(const ConstVectorRef &x) const {
    return left()(right()(x));
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const {
    MatrixXs Jleft = left().computeJacobian(right()(x));
    Jout.noalias() = Jleft * right().computeJacobian(x);
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
auto compose(const shared_ptr<C2FunctionTpl<Scalar>> &left,
             const shared_ptr<C2FunctionTpl<Scalar>> &right) {
  return std::make_shared<ComposeFunctionTpl<Scalar>>(left, right);
}

} // namespace nlp
} // namespace proxsuite

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/function-ops.txx"
#endif
