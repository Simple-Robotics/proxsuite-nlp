#pragma once

#include "proxnlp/function-base.hpp"
#include "proxnlp/function-ops.hpp"
#include "proxnlp/modelling/residuals/state-residual.hpp"

namespace proxnlp {

/**
 * @brief Linear residuals \f$r(x) = Ax + b\f$.
 */
template <typename _Scalar> struct LinearFunctionTpl : C2FunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  using Manifold = ManifoldAbstractTpl<Scalar>;

  LinearFunctionDifferenceToPoint(const shared_ptr<Manifold> &space,
                                  const ConstVectorRef &target,
                                  const ConstMatrixRef &A,
                                  const ConstVectorRef &b)
      : Base(std::make_shared<LinearFunctionTpl<Scalar>>(A, b),
             std::make_shared<ManifoldDifferenceToPoint<Scalar>>(space,
                                                                 target)) {
    PROXNLP_DIM_CHECK(target, space->nx());
  }
};

} // namespace proxnlp
