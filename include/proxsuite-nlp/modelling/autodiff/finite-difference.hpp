/// @file   Helper structs to define derivatives on a function
///         through finite differences.
#pragma once

#include "proxsuite-nlp/function-base.hpp"

namespace proxsuite {
namespace nlp {
namespace autodiff {

enum FDLevel {
  TOC1 = 0, ///< Cast to a \f$C^1\f$ function.
  TOC2 = 1  ///< Cast to a \f$C^2\f$ function.
};

/// Type of finite differences: forward, central, or forward.
///
enum FDType {
  BACKWARD, ///< Backward finite differences\f$\frac{f_{i} - f_{i-1}}h\f$
  CENTRAL,  ///< Central finite differences\f$\frac{f_{i+1} - f_{i-1}}h\f$
  FORWARD   ///< Forward finite differences\f$\frac{f_{i+1} - f_i}h\f$
};

namespace internal {

template <typename Scalar> struct finite_difference_impl {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  static void computeJacobian(const ManifoldAbstractTpl<Scalar> &space,
                              const BaseFunctionTpl<Scalar> &func,
                              const Scalar fd_eps, const ConstVectorRef &x,
                              MatrixRef Jout) {
    VectorXs ei(func.ndx());
    VectorXs xplus = space.neutral();
    VectorXs xminus = space.neutral();
    ei.setZero();
    for (int i = 0; i < func.ndx(); i++) {
      ei(i) = fd_eps;
      space.integrate(x, ei, xplus);
      space.integrate(x, -ei, xminus);
      Jout.col(i) = (func(xplus) - func(xminus)) / (2 * fd_eps);
      ei(i) = 0.;
    }
  }

  static void vectorHessianProduct(const ManifoldAbstractTpl<Scalar> &space,
                                   const C1FunctionTpl<Scalar> &func,
                                   const Scalar fd_eps, const ConstVectorRef &x,
                                   const ConstVectorRef &v, MatrixRef Hout) {
    VectorXs ei(func.ndx());
    VectorXs xplus = space.neutral();
    VectorXs xminus = space.neutral();
    MatrixXs Jplus(func.nr(), func.ndx());
    MatrixXs Jminus(func.nr(), func.ndx());
    Jplus.setZero();
    Jminus.setZero();
    ei.setZero();

    for (int i = 0; i < func.ndx(); i++) {
      ei(i) = fd_eps;
      space.integrate(x, ei, xplus);
      space.integrate(x, -ei, xminus);
      func.computeJacobian(xplus, Jplus);
      func.computeJacobian(xminus, Jminus);
      Hout.col(i) = ((Jplus - Jminus) / (2 * fd_eps)).transpose() * v;
      ei(i) = 0.;
    }
  }
};

} // namespace internal

template <typename Scalar, FDLevel n = TOC1> struct finite_difference_wrapper;

/** @brief    Approximate the derivatives of a given function using finite
 * differences, to downcast the function to a C1FunctionTpl.
 */
template <typename _Scalar>
struct finite_difference_wrapper<_Scalar, TOC1> : C1FunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  using Base = C1FunctionTpl<_Scalar>;
  using FuncType = BaseFunctionTpl<Scalar>;

  const ManifoldAbstractTpl<Scalar> &space;
  const FuncType &func;
  Scalar fd_eps;

  using Base::computeJacobian;

  finite_difference_wrapper(const ManifoldAbstractTpl<Scalar> &space,
                            const FuncType &func, const Scalar fd_eps)
      : Base(space, func.nr()), space(space), func(func), fd_eps(fd_eps) {}

  VectorXs operator()(const ConstVectorRef &x) const override {
    return func(x);
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const override {
    return internal::finite_difference_impl<Scalar>::computeJacobian(
        space, func, fd_eps, x, Jout);
  }
};

/** @brief    Approximate the second derivatives of a given function using
 * finite differences.
 *
 *  @details  This class inherits from the C1 finite_difference_wrapper<_Scalar,
 * TOC1>, and the C2 implementation.
 */
template <typename _Scalar>
struct finite_difference_wrapper<_Scalar, TOC2> : C2FunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  using Base = C2FunctionTpl<_Scalar>;
  using FuncType = C1FunctionTpl<Scalar>;

  const ManifoldAbstractTpl<Scalar> &space;
  const FuncType &func;
  Scalar fd_eps;

  using Base::computeJacobian;

  finite_difference_wrapper(const ManifoldAbstractTpl<Scalar> &space,
                            const FuncType &func, const Scalar fd_eps)
      : Base(space, func.nr()), space(space), func(func), fd_eps(fd_eps) {}

  VectorXs operator()(const ConstVectorRef &x) const override { func(x); }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const override {
    func.computeJacobian(x, Jout);
  }

  void vectorHessianProduct(const ConstVectorRef &x, const ConstVectorRef &v,
                            MatrixRef Hout) const override {
    internal::finite_difference_impl<Scalar>::vectorHessianProduct(
        space, func, fd_eps, x, v, Hout);
  }
};

} // namespace autodiff
} // namespace nlp
} // namespace proxsuite
