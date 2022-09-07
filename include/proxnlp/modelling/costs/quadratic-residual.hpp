#pragma once

#include "proxnlp/cost-function.hpp"
#include "proxnlp/function-base.hpp"

namespace proxnlp {

/**
 * @brief Quadratic function \f$\frac{1}{2} r^\top Qr + b^\top r + c\f$ of a
 * residual.
 *
 * Cost function which is a quadratic function
 * \f[
 *    \frac12 r(x)^\top Q r(x) + b^\top r(x) + c
 * \f]
 * of a residual function \f$r :\calX\to \RR^p\f$.
 */
template <typename _Scalar>
struct QuadraticResidualCost : public CostFunctionBaseTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  using FunctionType = C2FunctionTpl<Scalar>; // base constraint func to use
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostFunctionBaseTpl<Scalar>;
  using Base::computeGradient;
  using Base::computeHessian;

  /// Residual function \f$r(x)\f$ the composite cost is constructed over.
  shared_ptr<FunctionType> m_residual;
  /// Weights \f$Q\f$
  MatrixXs m_weights;
  /// Slope \f$b\f$
  VectorXs m_slope;
  /// Constant term \f$c\f$
  Scalar m_constant;

  QuadraticResidualCost(const shared_ptr<FunctionType> &residual,
                        const ConstMatrixRef &weights,
                        const ConstVectorRef &slope,
                        const Scalar constant = Scalar(0.))
      : Base(residual->nx(), residual->ndx()), m_residual(residual),
        m_weights(weights), m_slope(slope), m_constant(constant) {}

  QuadraticResidualCost(const shared_ptr<FunctionType> &residual,
                        const ConstMatrixRef &weights,
                        const Scalar constant = Scalar(0.))
      : QuadraticResidualCost(residual, weights, VectorXs::Zero(residual->nr()),
                              constant) {}

  template <typename... ResidualArgs>
  QuadraticResidualCost(const ConstMatrixRef &weights,
                        const ConstVectorRef &slope, const Scalar constant,
                        ResidualArgs &...args)
      : QuadraticResidualCost(new FunctionType(args...), weights, slope,
                              constant) {}

  Scalar call(const ConstVectorRef &x) const {
    VectorXs err = (*m_residual)(x);
    return Scalar(0.5) * err.dot(m_weights * err) + err.dot(m_slope) +
           m_constant;
  }

  void computeGradient(const ConstVectorRef &x, VectorRef out) const {
    MatrixXs Jres(m_residual->nr(), this->ndx());
    m_residual->computeJacobian(x, Jres);
    VectorXs err = (*m_residual)(x);
    out = Jres.transpose() * (m_weights * err + m_slope);
  }

  void computeHessian(const ConstVectorRef &x, MatrixRef out) const {
    MatrixXs Jres(m_residual->nr(), this->ndx());
    m_residual->computeJacobian(x, Jres);
    VectorXs err = (*m_residual)(x);
    out = Jres.transpose() * (m_weights * Jres) +
          m_residual->vectorHessianProduct(x, m_weights * err + m_slope);
  }
};

} // namespace proxnlp
