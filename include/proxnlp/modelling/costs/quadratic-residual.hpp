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
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
  using Base = CostFunctionBaseTpl<Scalar>;
  using Base::computeGradient;
  using Base::computeHessian;
  using FunctionPtr = shared_ptr<FunctionType>;

  /// Residual function \f$r(x)\f$ the composite cost is constructed over.
  FunctionPtr residual_;
  /// Weights \f$Q\f$
  MatrixXs weights_;
  /// Slope \f$b\f$
  VectorXs slope_;
  /// Constant term \f$c\f$
  Scalar constant_;

  QuadraticResidualCost(FunctionPtr residual, const ConstMatrixRef &weights,
                        const ConstVectorRef &slope,
                        const Scalar constant = Scalar(0.))
      : Base(residual->nx(), residual->ndx()), residual_(residual),
        weights_(weights), slope_(slope), constant_(constant),
        err(residual_->nr()), Jres(residual_->nr(), this->ndx()),
        JtW(this->ndx(), residual_->nr()), H(this->ndx(), this->ndx()) {
    err.setZero();
    tmp_w_err = err;
    H.setZero();
  }

  QuadraticResidualCost(FunctionPtr residual, const ConstMatrixRef &weights,
                        const Scalar constant = Scalar(0.))
      : QuadraticResidualCost(residual, weights, VectorXs::Zero(residual->nr()),
                              constant) {}

  template <typename... ResidualArgs>
  QuadraticResidualCost(const ConstMatrixRef &weights,
                        const ConstVectorRef &slope, const Scalar constant,
                        ResidualArgs &...args)
      : QuadraticResidualCost(std::make_shared<FunctionType>(args...), weights,
                              slope, constant) {}

  Scalar call(const ConstVectorRef &x) const {
    auto &self = const_cast_self();
    self.err = (*residual_)(x);

    PROXNLP_NOMALLOC_BEGIN;

    self.tmp_w_err.noalias() = weights_ * err;
    Scalar res = Scalar(0.5) * err.dot(tmp_w_err) + err.dot(slope_) + constant_;

    PROXNLP_NOMALLOC_END;

    return res;
  }

  void computeGradient(const ConstVectorRef &x, VectorRef out) const {
    auto &self = const_cast_self();
    residual_->computeJacobian(x, self.Jres);

    self.JtW.noalias() = Jres.transpose() * weights_;
    out.noalias() = JtW * err;
    out.noalias() += Jres.transpose() * slope_;
  }

  void computeHessian(const ConstVectorRef &x, MatrixRef out) const {
    auto &self = const_cast_self();
    self.tmp_w_err.noalias() = weights_ * err;
    self.tmp_w_err += slope_;

    residual_->vectorHessianProduct(x, self.tmp_w_err, self.H);
    out = H;

    residual_->computeJacobian(x, self.Jres);
    out.noalias() += JtW * Jres;
  }

protected:
  QuadraticResidualCost &const_cast_self() const {
    return const_cast<QuadraticResidualCost &>(*this);
  }

private:
  VectorXs err;
  VectorXs tmp_w_err;
  MatrixXs Jres;
  RowMatrixXs JtW;
  MatrixXs H;
};

} // namespace proxnlp

#ifdef PROXNLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxnlp/modelling/costs/quadratic-residual.txx"
#endif
