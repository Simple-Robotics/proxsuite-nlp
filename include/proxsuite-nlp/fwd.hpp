/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
/// @brief     Forward declarations and configuration macros.
#pragma once

#ifdef EIGEN_DEFAULT_IO_FORMAT
#undef EIGEN_DEFAULT_IO_FORMAT
#endif
#define EIGEN_DEFAULT_IO_FORMAT                                                \
  Eigen::IOFormat(Eigen::StreamPrecision, 0, ",", "\n", "[", "]")

#include <memory>

/// @brief  Main package namespace.
///
/// A primal-dual augmented Lagrangian-type solver and its utilities
/// (e.g. modelling, memory management, helpers...)
namespace proxnlp {

/// Automatic differentiation utilities.
namespace autodiff {}

/// Helper functions and structs.
namespace helpers {}

/// Use the STL shared_ptr.
using std::shared_ptr;
/// Use the STL unique_ptr.
using std::unique_ptr;

} // namespace proxnlp

#include "proxsuite-nlp/math.hpp"
#include "proxsuite-nlp/exceptions.hpp"
#include "proxsuite-nlp/macros.hpp"
#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/deprecated.hpp"
#include "proxsuite-nlp/warning.hpp"

namespace proxnlp {

template <typename T, typename... Args>
auto allocate_shared_eigen_aligned(Args &&...args) {
  return std::allocate_shared<T>(Eigen::aligned_allocator<T>(),
                                 std::forward<Args>(args)...);
}

// fwd BCLParams
template <typename Scalar> struct BCLParamsTpl;

/* Function types */

// fwd BaseFunction
template <typename Scalar> struct BaseFunctionTpl;

// fwd C1FunctionTpl
template <typename Scalar> struct C1FunctionTpl;

// fwd C2FunctionTpl
template <typename Scalar> struct C2FunctionTpl;

// fwd func_to_cost
template <typename Scalar> struct func_to_cost;

// fwd ComposeFunctionTpl
template <typename Scalar> struct ComposeFunctionTpl;

template <typename Scalar>
auto compose(const shared_ptr<C2FunctionTpl<Scalar>> &left,
             const shared_ptr<C2FunctionTpl<Scalar>> &right);

// fwd Cost
template <typename Scalar> struct CostFunctionBaseTpl;

/* Manifolds */

// fwd ManifoldAbstractTpl
template <typename Scalar, int Options = 0> struct ManifoldAbstractTpl;

template <typename Scalar, int Dim = Eigen::Dynamic, int Options = 0>
struct VectorSpaceTpl;

template <typename Scalar> struct CartesianProductTpl;

template <typename Base> struct TangentBundleTpl;

// fwd ConstraintSetBase
template <typename Scalar> struct ConstraintSetBase;

// fwd ConstraintObject
template <typename Scalar> struct ConstraintObjectTpl;

/* Solver structs */

// fwd Problem
template <typename Scalar> struct ProblemTpl;

// fwd ResultsTpl
template <typename Scalar> struct ResultsTpl;

// fwd WorkspaceTpl
template <typename Scalar> struct WorkspaceTpl;

/// Verbosity level.
enum VerboseLevel { QUIET = 0, VERBOSE = 1, VERYVERBOSE = 2 };

template <typename Scalar> class ProxNLPSolverTpl;

} // namespace proxnlp
