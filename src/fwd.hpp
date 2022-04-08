#pragma once

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
};

// fwd BaseFunction
template<typename Scalar>
struct BaseFunction;

// fwd C1Function
template<typename Scalar>
struct C1Function;

// fwd C2Function
template<typename Scalar>
struct C2Function;

// fwd ComposeFunction
template<typename Scalar>
struct ComposeFunction;

// fwd ManifoldAbstractTpl
template<typename Scalar, int Options=0>
struct ManifoldAbstractTpl;

template<typename Base>
struct TangentBundleTpl;

// fwd Cost
template<typename Scalar>
struct CostFunctionBase;

// fwd ConstraintSetBase
template<typename Scalar>
struct ConstraintSetBase;

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
