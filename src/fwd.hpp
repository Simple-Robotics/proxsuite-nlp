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
  using VectorMap = Eigen::Map<VectorXs>;
  using VectorOfVectors = std::vector<VectorXs>;
  using VectorOfVecMap = std::vector<VectorMap>;
  using VectorRef = Eigen::Ref<VectorXs>;
  using VectorOfRef = std::vector<VectorRef>;
  using MatrixRef = Eigen::Ref<MatrixXs>;
  using ConstVectorRef = Eigen::Ref<const VectorXs>;
  using ConstMatrixRef = Eigen::Ref<const MatrixXs>;
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
  
  /// Shorthand for the infinity norm
  /// code from proxqp
  template<typename MatType>
  typename MatType::Scalar
  infNorm(const Eigen::MatrixBase<MatType>& z)
  {
    if (z.rows() == 0 || z.cols() == 0)
    {
      return 0.;
    } else {
      return z.template lpNorm<Eigen::Infinity>();
    }
  }

} // namespace math

}  // namespace lienlp
