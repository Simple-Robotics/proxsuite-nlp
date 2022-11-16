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
        weights_(weights), slope_(slope), constant_(constant),
        err(residual_->nr()), Jres(residual_->nr(), this->ndx()),
        JtW(this->ndx(), residual_->nr()), H(this->ndx(), this->ndx()) {
    err.setZero();
    tmp_w_err = err;
    H.setZero();
  }

  QuadraticResidualCost(const shared_ptr<FunctionType> &residual,
                        const ConstMatrixRef &weights,
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
    PROXNLP_EIGEN_CONST_CAST(VectorXs, err) = (*residual_)(x);

    PROXNLP_EIGEN_ALLOW_MALLOC(false);

    PROXNLP_EIGEN_CONST_CAST(VectorXs, tmp_w_err).noalias() = weights_ * err;
    Scalar res = Scalar(0.5) * err.dot(tmp_w_err) + err.dot(slope_) + constant_;

    PROXNLP_EIGEN_ALLOW_MALLOC(true);

    return res;
  }

  void computeGradient(const ConstVectorRef &x, VectorRef out) const {
    MatrixXs &Jres_mut = PROXNLP_EIGEN_CONST_CAST(MatrixXs, Jres);
    RowMatrixXs &JtW_mut = PROXNLP_EIGEN_CONST_CAST(RowMatrixXs, JtW);
    residual_->computeJacobian(x, Jres_mut);

    JtW_mut.noalias() = Jres_mut.transpose() * weights_;
    out.noalias() = JtW_mut * err;
    out.noalias() += Jres.transpose() * slope_;
  }

  void computeHessian(const ConstVectorRef &x, MatrixRef out) const {
    MatrixXs &Jres_mut = PROXNLP_EIGEN_CONST_CAST(MatrixXs, Jres);
    residual_->computeJacobian(x, Jres_mut);
    PROXNLP_EIGEN_CONST_CAST(VectorXs, err) = (*residual_)(x);

    out.noalias() = JtW * Jres_mut;
    VectorXs &tmp_mut = PROXNLP_EIGEN_CONST_CAST(VectorXs, tmp_w_err);
    tmp_mut.noalias() = weights_ * err;
    tmp_mut += slope_;

    MatrixXs &H_mut = PROXNLP_EIGEN_CONST_CAST(MatrixXs, H);
    residual_->vectorHessianProduct(x, tmp_mut, H_mut);
    out += H_mut;
  }

private:
  VectorXs err;
  VectorXs tmp_w_err;
  MatrixXs Jres;
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
  RowMatrixXs JtW;
  MatrixXs H;
};

} // namespace proxnlp
