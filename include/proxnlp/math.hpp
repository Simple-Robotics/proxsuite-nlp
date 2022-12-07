/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/exceptions.hpp"
#include <Eigen/Core>

#include <cmath>
#include <type_traits>
#include <limits>
#include <vector>

#define PROXNLP_RAISE_IF_NAN(value)                                            \
  if (::proxnlp::math::check_value(value))                                     \
  PROXNLP_RUNTIME_ERROR("Encountered NaN.\n")

#define PROXNLP_RAISE_IF_NAN_NAME(value, name)                                 \
  if (::proxnlp::math::check_value(value))                                     \
  PROXNLP_RUNTIME_ERROR(fmt::format("Encountered NaN for value {:s}.\n", name))

/// Macro typedefs for dynamic-sized vectors/matrices, used for cost funcs,
/// merit funcs because we don't CRTP them and virtual members funcs can't be
/// templated.
#define PROXNLP_DYNAMIC_TYPEDEFS(Scalar)                                       \
  using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;                   \
  using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;      \
  using VectorOfVectors = std::vector<VectorXs>;                               \
  using VectorRef = Eigen::Ref<VectorXs>;                                      \
  using MatrixRef = Eigen::Ref<MatrixXs>;                                      \
  using ConstVectorRef = Eigen::Ref<const VectorXs>;                           \
  using ConstMatrixRef = Eigen::Ref<const MatrixXs>;                           \
  using VectorOfRef = std::vector<VectorRef>;                                  \
  using Vector3s = Eigen::Matrix<Scalar, 3, 1>;                                \
  using Vector6s = Eigen::Matrix<Scalar, 6, 1>;                                \
  using Matrix3Xs = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;                  \
  using Matrix6Xs = Eigen::Matrix<Scalar, 6, Eigen::Dynamic>;                  \
  using Matrix6s = Eigen::Matrix<Scalar, 6, 6>

namespace proxnlp {

/** @brief  Typedefs for math (Eigen vectors, matrices) depending on scalar
 * type.
 *
 */
template <typename _Scalar> struct math_types {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(_Scalar);
};

/// Math utilities
namespace math {
template <typename MatType>
typename MatType::Scalar infty_norm(const Eigen::MatrixBase<MatType> &z) {
  if (z.rows() == 0 || z.cols() == 0) {
    return 0.;
  } else {
    return z.template lpNorm<Eigen::Infinity>();
  }
}

template <typename MatType>
typename MatType::Scalar infty_norm(const std::vector<MatType> &z) {
  const std::size_t n = z.size();
  typename MatType::Scalar out = 0.;
  for (std::size_t i = 0; i < n; i++) {
    out = std::max(out, infty_norm(z[i]));
  }
  return out;
}

/// @brief Check that a scalar is neither inf, nor NaN.
template <typename Scalar> inline bool check_scalar(const Scalar value) {
  return std::isnan(value) || std::isinf(value);
}

/**
 * @brief Tests whether @p a and @p b are close, within absolute and relative
 * precision @p prec.
 */
template <typename Scalar>
bool scalar_close(const Scalar a, const Scalar b,
                  const Scalar prec = std::numeric_limits<Scalar>::epsilon()) {
  return std::abs(a - b) < prec * (1 + std::max(std::abs(a), std::abs(b)));
}

template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
bool check_value(const T &x) {
  static_assert(std::is_scalar<T>::value, "Parameter T should be scalar.");
  return check_scalar(x);
}

template <typename MatrixType>
bool check_value(const Eigen::MatrixBase<MatrixType> &x) {
  return (x.hasNaN() || (!x.allFinite()));
}

} // namespace math

} // namespace proxnlp
