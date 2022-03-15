#pragma once

#include <Eigen/Core>

#include <vector>
#include <memory>

namespace lienlp {

/// Use the STL shared_ptr.
using std::shared_ptr;

/// Base trait struct for CRTP.
template<class C>
struct traits {};

template<typename _Scalar>
struct math_types
{
  using Scalar = _Scalar;
  using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorOfVectors = std::vector<VectorXs>;
};

}  // namespace lienlp
