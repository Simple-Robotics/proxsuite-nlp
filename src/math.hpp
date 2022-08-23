#pragma once

#include <Eigen/Core>

#include <vector>

namespace proxnlp {

/** @brief  Typedefs for math (Eigen vectors, matrices) depending on scalar
 * type.
 *
 */
template <typename _Scalar> struct math_types {
  using Scalar = _Scalar;
  using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorOfVectors = std::vector<VectorXs>;
  using VectorRef = Eigen::Ref<VectorXs>;
  using MatrixRef = Eigen::Ref<MatrixXs>;
  using ConstVectorRef = Eigen::Ref<const VectorXs>;
  using ConstMatrixRef = Eigen::Ref<const MatrixXs>;
  using VectorOfRef = std::vector<VectorRef>;

  using Vector3s = Eigen::Matrix<Scalar, 3, 1>;
  using Vector6s = Eigen::Matrix<Scalar, 6, 1>;

  using Matrix3Xs = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
  using Matrix6Xs = Eigen::Matrix<Scalar, 6, Eigen::Dynamic>;
  using Matrix6s = Eigen::Matrix<Scalar, 6, 6>;
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

} // namespace math

} // namespace proxnlp
