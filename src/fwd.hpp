/** 
 * @file    Forward declarations and configuration macros.
 */
#pragma once

#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(2, 0, ", ")

#include <Eigen/Core>

#include <vector>
#include <memory>

/**
 * Main namespace of the package.
 */
namespace lienlp
{

/// Use the STL shared_ptr.
using std::shared_ptr;

/** @brief  Typedefs for math (Eigen vectors, matrices) depending on scalar type.
 * 
 */
template<typename _Scalar>
struct math_types
{
  using Scalar = _Scalar;
  using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorOfVectors = std::vector<VectorXs>;
  using VectorRef = Eigen::Ref<VectorXs>;
  using MatrixRef = Eigen::Ref<MatrixXs>;
  using ConstVectorRef = Eigen::Ref<const VectorXs>;
  using ConstMatrixRef = Eigen::Ref<const MatrixXs>;
  using VectorOfRef = std::vector<VectorRef>;
};

/* Function types */

// fwd BaseFunction
template<typename Scalar>
struct BaseFunctionTpl;

// fwd C1FunctionTpl
template<typename Scalar>
struct C1FunctionTpl;

// fwd C2FunctionTpl
template<typename Scalar>
struct C2FunctionTpl;

// fwd ComposeFunctionTpl
template<typename Scalar>
struct ComposeFunctionTpl;

// fwd Cost
template<typename Scalar>
struct CostFunctionBaseTpl;

/* Manifolds */

// fwd ManifoldAbstractTpl
template<typename Scalar, int Options=0>
struct ManifoldAbstractTpl;

template<typename Base>
struct TangentBundleTpl;

// fwd ConstraintSetBase
template<typename Scalar>
struct ConstraintSetBase;

/* Solver structs */

// fwd Problem
template<typename Scalar>
struct ProblemTpl;

// fwd SResults
template<typename Scalar>
struct SResults;

// fwd Workspace
template<typename Scalar>
struct SWorkspace;

template<typename Scalar>
class SolverTpl;

/// Math utils
namespace math
{
  template<typename MatType>
  typename MatType::Scalar infty_norm(const Eigen::MatrixBase<MatType>& z)
  {
    if (z.rows() == 0 || z.cols() == 0)
    {
      return 0.;
    } else {
      return z.template lpNorm<Eigen::Infinity>();
    }
  }

  template<typename MatType>
  typename MatType::Scalar infty_norm(const std::vector<Eigen::MatrixBase<MatType>>& z)
  {
    const std::size_t n = z.size();
    typename MatType::Scalar out = 0.;
    for (std::size_t i = 0; i < n; i++)
    {
      out = std::max(out, infty_norm(z));
    }
    return out;
  }

} // namespace math

}  // namespace lienlp
