/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include <fmt/ostream.h>

/// Specialize fmt::formatter using the operator<< implementation for Eigen
/// types.

template <typename MatrixType>
struct fmt::formatter<
    MatrixType,
    std::enable_if_t<
        std::is_base_of_v<Eigen::DenseBase<MatrixType>, MatrixType>, char>>
    : fmt::ostream_formatter {};
