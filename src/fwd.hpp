#pragma once

#include <Eigen/Core>

#include <vector>
#include <memory>

/**
 * Main namespace of the package.
 */
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

// fwd Cost
template<typename Scalar>
struct CostFunctionBase;

// fwd Problem
template<typename Scalar>
struct Problem;

// fwd SResults
template<typename Scalar>
struct SResults;

/// Shorthand for the infinity norm
/// code from proxqp
template<typename Mat_t>
typename Mat_t::Scalar
infNorm(const Eigen::MatrixBase<Mat_t>& z)
{
  if (z.rows() == 0 || z.cols() == 0)
  {
    return typename Mat_t::Scalar(0);
  } else {
    return z.template lpNorm<Eigen::Infinity>();
  }
}

}  // namespace lienlp
