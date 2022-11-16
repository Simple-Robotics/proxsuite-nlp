/** @file Base definitions for function classes.
 */
#pragma once

#include "proxnlp/fwd.hpp"

namespace proxnlp {
/**
 * @brief Base function type.
 */
template <typename _Scalar> struct BaseFunctionTpl : math_types<_Scalar> {
protected:
  int nx_;
  int ndx_;
  int nr_;

public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  BaseFunctionTpl(const int nx, const int ndx, const int nr)
      : nx_(nx), ndx_(ndx), nr_(nr) {}

  /// @brief      Evaluate the residual at a given point x.
  virtual VectorXs operator()(const ConstVectorRef &x) const = 0;

  virtual ~BaseFunctionTpl() = default;

  /// Get function input vector size (representation of manifold).
  int nx() const { return nx_; }
  /// Get input manifold's tangent space dimension.
  int ndx() const { return ndx_; }
  /// Get function codimension.
  int nr() const { return nr_; }
};

/** @brief  Differentiable function, with method for the Jacobian.
 */
template <typename _Scalar>
struct C1FunctionTpl : public BaseFunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  using Base = BaseFunctionTpl<_Scalar>;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  C1FunctionTpl(const int nx, const int ndx, const int nr)
      : Base(nx, ndx, nr) {}

  Base &toBase() { return static_cast<Base &>(*this); }

  /// @brief      Jacobian matrix of the constraint function.
  virtual void computeJacobian(const ConstVectorRef &x,
                               MatrixRef Jout) const = 0;

  /** @copybrief computeJacobian()
   *
   * Allocated version of the computeJacobian() method.
   */
  MatrixXs computeJacobian(const ConstVectorRef &x) const {
    MatrixXs Jout(this->nr(), this->ndx());
    computeJacobian(x, Jout);
    return Jout;
  }
};

/** @brief  Twice-differentiable function, with method Jacobian and
 * vector-hessian product evaluation.
 */
template <typename _Scalar>
struct C2FunctionTpl : public C1FunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  using Base = C1FunctionTpl<_Scalar>;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  C2FunctionTpl(const int nx, const int ndx, const int nr)
      : Base(nx, ndx, nr) {}

  Base &toC1() { return static_cast<Base &>(*this); }

  /// @brief      Vector-hessian product.
  virtual void vectorHessianProduct(const ConstVectorRef &,
                                    const ConstVectorRef &,
                                    MatrixRef Hout) const {
    Hout.setZero();
  }
};

} // namespace proxnlp
