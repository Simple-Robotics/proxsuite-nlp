#pragma once

/// Macro typedefs for dynamic-sized vectors/matrices, used for cost funcs, merit funcs
/// because we don't CRTP them and virtual members funcs can't be templated.
#define LIENLP_DYNAMIC_TYPEDEFS(Scalar)                       \
  using VectorXs = typename math_types<Scalar>::VectorXs;     \
  using MatrixXs = typename math_types<Scalar>::MatrixXs;     \
  using VectorOfVectors = typename math_types<Scalar>::VectorOfVectors; \
  using VectorOfRef = typename math_types<Scalar>::VectorOfRef;   \
  using VectorRef = typename math_types<Scalar>::VectorRef;   \
  using MatrixRef = typename math_types<Scalar>::MatrixRef;   \
  using ConstVectorRef = typename math_types<Scalar>::ConstVectorRef;   \
  using ConstMatrixRef = typename math_types<Scalar>::ConstMatrixRef;

/// @brief Macro empty arg
#define LIENLP_MACRO_EMPTY_ARG

#define LIENLP_EIGEN_CONST_CAST(type, obj) const_cast<type &>(obj)

#define LIENLP_FUNCTOR_TYPEDEFS(Scalar)          \
  LIENLP_DYNAMIC_TYPEDEFS(Scalar)                \
  using ReturnType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;   \
  using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
