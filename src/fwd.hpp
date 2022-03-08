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

/// Macro typedefs for dynamic-sized vectors/matrices, used for cost funcs, merit funcs
/// because we don't CRTP them and virtual members funcs can't be templated.
#define LIENLP_DEFINE_DYNAMIC_TYPES(_Scalar)                \
  using Scalar = _Scalar;                                   \
  using VectorXs = typename math_types<Scalar>::VectorXs;   \
  using MatrixXs = typename math_types<Scalar>::MatrixXs;   \
  using RefVector = Eigen::Ref<const VectorXs>;             \
  using RefMatrix = Eigen::Ref<const MatrixXs>;

}  // namespace lienlp
