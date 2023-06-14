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
struct QuadraticResidualCostTpl : public CostFunctionBaseTpl<_Scalar> {
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
  bool gauss_newton_;

  QuadraticResidualCostTpl(FunctionPtr residual, const ConstMatrixRef &weights,
                           const ConstVectorRef &slope,
                           const Scalar constant = Scalar(0.));

  QuadraticResidualCostTpl(FunctionPtr residual, const ConstMatrixRef &weights,
                           const Scalar constant = Scalar(0.))
      : QuadraticResidualCostTpl(residual, weights,
                                 VectorXs::Zero(residual->nr()), constant) {}

  /// @brief Constructor using the template parameter as the underlying type of
  /// the residual.
  template <typename Underlying, typename... ResidualArgs>
  QuadraticResidualCostTpl(const ConstMatrixRef &weights,
                           const ConstVectorRef &slope, const Scalar constant,
                           ResidualArgs &...args)
      : QuadraticResidualCostTpl(std::make_shared<Underlying>(args...), weights,
                                 slope, constant) {}

  Scalar call(const ConstVectorRef &x) const;

  void computeGradient(const ConstVectorRef &x, VectorRef out) const;

  void computeHessian(const ConstVectorRef &x, MatrixRef out) const;

protected:
  mutable VectorXs err;
  mutable VectorXs tmp_w_err;
  mutable MatrixXs Jres;
  mutable RowMatrixXs JtW;
  mutable MatrixXs H;
};

} // namespace proxnlp

#include "proxnlp/modelling/costs/quadratic-residual.hxx"

#ifdef PROXNLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxnlp/modelling/costs/quadratic-residual.txx"
#endif
