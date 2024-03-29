#pragma once

#include "./quadratic-residual.hpp"

namespace proxsuite {
namespace nlp {

template <typename Scalar>
QuadraticResidualCostTpl<Scalar>::QuadraticResidualCostTpl(
    FunctionPtr residual, const ConstMatrixRef &weights,
    const ConstVectorRef &slope, const Scalar constant)
    : Base(residual->nx(), residual->ndx()), residual_(residual),
      weights_(weights), slope_(slope), constant_(constant),
      gauss_newton_(true), err(residual_->nr()),
      Jres(residual_->nr(), this->ndx()), JtW(this->ndx(), residual_->nr()),
      H(this->ndx(), this->ndx()) {
  Jres.setZero();
  err.setZero();
  tmp_w_err = err;
  H.setZero();
}

template <typename Scalar>
Scalar QuadraticResidualCostTpl<Scalar>::call(const ConstVectorRef &x) const {
  err = (*residual_)(x);

  PROXSUITE_NLP_NOMALLOC_BEGIN;

  tmp_w_err.noalias() = weights_ * err;
  Scalar res = Scalar(0.5) * err.dot(tmp_w_err) + err.dot(slope_) + constant_;

  PROXSUITE_NLP_NOMALLOC_END;

  return res;
}

template <typename Scalar>
void QuadraticResidualCostTpl<Scalar>::computeGradient(const ConstVectorRef &x,
                                                       VectorRef out) const {
  residual_->computeJacobian(x, Jres);

  JtW.noalias() = Jres.transpose() * weights_;
  out.noalias() = JtW * err;
  out.noalias() += Jres.transpose() * slope_;
}

template <typename Scalar>
void QuadraticResidualCostTpl<Scalar>::computeHessian(const ConstVectorRef &x,
                                                      MatrixRef out) const {
  tmp_w_err.noalias() = weights_ * err;
  tmp_w_err += slope_;

  if (!gauss_newton_) {
    residual_->vectorHessianProduct(x, tmp_w_err, H);
    out = H;
  } else {
    out.setZero();
  }

  Jres.setZero();
  residual_->computeJacobian(x, Jres);
  JtW.noalias() = Jres.transpose() * weights_;
  out.noalias() += JtW * Jres;
}

} // namespace nlp
} // namespace proxsuite
