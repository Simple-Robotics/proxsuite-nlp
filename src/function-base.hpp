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
  const int m_nx;
  const int m_ndx;
  const int m_nr;

public:
  using Scalar = _Scalar;
  PROXNLP_FUNCTION_TYPEDEFS(Scalar);

  /// @brief      Evaluate the residual at a given point x.
  virtual ReturnType operator()(const ConstVectorRef &x) const = 0;

  BaseFunctionTpl(const int nx, const int ndx, const int nr)
      : m_nx(nx), m_ndx(ndx), m_nr(nr) {}

  virtual ~BaseFunctionTpl() = default;

  /// Get function input vector size (representation of manifold).
  int nx() const { return m_nx; }
  /// Get input manifold's tangent space dimension.
  int ndx() const { return m_ndx; }
  /// Get function codimension.
  int nr() const { return m_nr; }
};

/** @brief  Differentiable function, with method for the Jacobian.
 */
template <typename _Scalar>
struct C1FunctionTpl : public BaseFunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  using Base = BaseFunctionTpl<_Scalar>;
  PROXNLP_FUNCTION_TYPEDEFS(Scalar);

  Base &toBase() { return static_cast<Base &>(*this); }

  C1FunctionTpl(const int nx, const int ndx, const int nr)
      : Base(nx, ndx, nr) {}

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
  PROXNLP_FUNCTION_TYPEDEFS(Scalar);

  Base &toC1() { return static_cast<Base &>(*this); }

  C2FunctionTpl(const int nx, const int ndx, const int nr)
      : Base(nx, ndx, nr) {}

  /// @brief      Vector-hessian product.
  virtual void vectorHessianProduct(const ConstVectorRef &,
                                    const ConstVectorRef &,
                                    MatrixRef Hout) const {
    Hout.setZero();
  }

  MatrixXs vectorHessianProduct(const ConstVectorRef &x,
                                const ConstVectorRef &v) const {
    MatrixXs J(this->ndx(), this->ndx());
    vectorHessianProduct(x, v, J);
    return J;
  }
};

} // namespace proxnlp
