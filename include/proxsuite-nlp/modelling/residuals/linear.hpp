#pragma once

#include "proxsuite-nlp/function-base.hpp"
#include "proxsuite-nlp/function-ops.hpp"
#include "proxsuite-nlp/modelling/residuals/state-residual.hpp"
#include "proxsuite-nlp/third-party/polymorphic_cxx14.hpp"

namespace proxsuite {
namespace nlp {
using xyz::polymorphic;

/**
 * @brief Linear residuals \f$r(x) = Ax + b\f$.
 */
template <typename _Scalar> struct LinearFunctionTpl : C2FunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  using Base = C2FunctionTpl<Scalar>;
  using Base::computeJacobian;

  MatrixXs mat;
  VectorXs b;

  LinearFunctionTpl(const ConstMatrixRef &A, const ConstVectorRef &b)
      : Base((int)A.cols(), (int)A.cols(), (int)A.rows()), mat(A), b(b) {}

  LinearFunctionTpl(const ConstMatrixRef &A)
      : LinearFunctionTpl(A, VectorXs::Zero(A.rows())) {}

  VectorXs operator()(const ConstVectorRef &x) const { return mat * x + b; }

  void computeJacobian(const ConstVectorRef &, MatrixRef Jout) const {
    Jout = mat;
  }
};

/** @brief    Linear function of difference vector on a manifold, of the form
 *            \f$ r(x) = A(x \ominus \bar{x}) + b \f$.
 */
template <typename _Scalar>
struct LinearFunctionDifferenceToPoint : ComposeFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  using Base = ComposeFunctionTpl<Scalar>;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  using Manifold = ManifoldAbstractTpl<Scalar>;

  LinearFunctionDifferenceToPoint(const polymorphic<Manifold> &space,
                                  const ConstVectorRef &target,
                                  const ConstMatrixRef &A,
                                  const ConstVectorRef &b)
      : Base(std::make_shared<LinearFunctionTpl<Scalar>>(A, b),
             std::make_shared<ManifoldDifferenceToPoint<Scalar>>(space,
                                                                 target)) {
    PROXSUITE_NLP_DIM_CHECK(target, space->nx());
  }
};

} // namespace nlp
} // namespace proxsuite
