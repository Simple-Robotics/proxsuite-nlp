/** 
 * @file    Forward declarations and configuration macros.
 */
#pragma once

#ifdef EIGEN_DEFAULT_IO_FORMAT
#undef EIGEN_DEFAULT_IO_FORMAT
#endif
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, ",", "\n", "[", "]")

#include <memory>


/// @brief  Main package namespace.
///
/// A primal-dual augmented Lagrangian-type solver and its utilities
/// (e.g. modelling, memory management, helpers...)
namespace proxnlp
{

/// Automatic differentiation utilities.
namespace autodiff {}

/// Helper functions and structs.
namespace helpers {}

/// Use the STL shared_ptr.
using std::shared_ptr;

} // namespace proxnlp

#include "proxnlp/math.hpp"
#include "proxnlp/macros.hpp"
#include "proxnlp/config.hpp"

namespace proxnlp
{

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

template<typename Scalar, int Dim=Eigen::Dynamic, int Options=0>
struct VectorSpaceTpl;

template<typename Base>
struct TangentBundleTpl;

// fwd ConstraintObject
template<typename Scalar>
struct ConstraintSetBase;

// fwd ConstraintObject
template<typename Scalar>
struct ConstraintObject;

/* Solver structs */

// fwd Problem
template<typename Scalar>
struct ProblemTpl;

// fwd ResultsTpl
template<typename Scalar>
struct ResultsTpl;

// fwd WorkspaceTpl
template<typename Scalar>
struct WorkspaceTpl;

/// Verbosity level.
enum VerboseLevel
{
  QUIET=0,
  VERBOSE=1,
  VERY=2
};

template<typename Scalar>
class SolverTpl;

}  // namespace proxnlp
