#pragma once

/// Macro typedefs for dynamic-sized vectors/matrices, used for cost funcs,
/// merit funcs because we don't CRTP them and virtual members funcs can't be
/// templated.
#define PROXNLP_DYNAMIC_TYPEDEFS(Scalar)                                       \
  using VectorXs = typename math_types<Scalar>::VectorXs;                      \
  using MatrixXs = typename math_types<Scalar>::MatrixXs;                      \
  using VectorOfVectors = typename math_types<Scalar>::VectorOfVectors;        \
  using VectorOfRef = typename math_types<Scalar>::VectorOfRef;                \
  using VectorRef = typename math_types<Scalar>::VectorRef;                    \
  using MatrixRef = typename math_types<Scalar>::MatrixRef;                    \
  using ConstVectorRef = typename math_types<Scalar>::ConstVectorRef;          \
  using ConstMatrixRef = typename math_types<Scalar>::ConstMatrixRef

/// @brief Macro empty arg
#define PROXNLP_MACRO_EMPTY_ARG

#define PROXNLP_EIGEN_CONST_CAST(type, obj) const_cast<type &>(obj)

#define PROXNLP_FUNCTION_TYPEDEFS(Scalar)                                      \
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);                                            \
  using ReturnType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
