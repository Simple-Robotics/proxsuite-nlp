/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include <fmt/ostream.h>
#include <fmt/ranges.h>

/// Specialize fmt::formatter using the operator<< implementation for Eigen
/// types.
template <typename MatrixType>
struct fmt::formatter<MatrixType,
                      proxsuite::nlp::enable_if_eigen_dense<MatrixType, char>>
    : fmt::ostream_formatter {};

template <typename MatrixType>
struct fmt::is_range<MatrixType,
                     proxsuite::nlp::enable_if_eigen_dense<MatrixType, char>>
    : std::false_type {};
