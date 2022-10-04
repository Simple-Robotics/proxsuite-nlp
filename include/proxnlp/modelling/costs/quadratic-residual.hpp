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
  shared_ptr<FunctionType> residual_;
  /// Weights \f$Q\f$
  MatrixXs weights_;
  /// Slope \f$b\f$
  VectorXs slope_;
  /// Constant term \f$c\f$
  Scalar constant_;

  QuadraticResidualCost(const shared_ptr<FunctionType> &residual,
                        const ConstMatrixRef &weights,
                        const ConstVectorRef &slope,
                        const Scalar constant = Scalar(0.))
      : Base(residual->nx(), residual->ndx()), residual_(residual),
        weights_(weights), slope_(slope), constant_(constant) {}

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
    VectorXs err = (*residual_)(x);
    return Scalar(0.5) * err.dot(weights_ * err) + err.dot(slope_) + constant_;
  }

  void computeGradient(const ConstVectorRef &x, VectorRef out) const {
    MatrixXs Jres(residual_->nr(), this->ndx());
    residual_->computeJacobian(x, Jres);
    VectorXs err = (*residual_)(x);
    out = Jres.transpose() * (weights_ * err + slope_);
  }

  void computeHessian(const ConstVectorRef &x, MatrixRef out) const {
    MatrixXs Jres(residual_->nr(), this->ndx());
    residual_->computeJacobian(x, Jres);
    VectorXs err = (*residual_)(x);
    out = Jres.transpose() * (weights_ * Jres) +
          residual_->vectorHessianProduct(x, weights_ * err + slope_);
  }
};

} // namespace proxnlp
