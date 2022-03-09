#pragma once

#include <Eigen/Core>

namespace lienlp {

/// Base trait struct for CRTP.
template<class C>
struct traits {};

template<typename _Scalar>
struct math_types
{
  using Scalar = _Scalar;
  using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
};

}  // namespace lienlp
