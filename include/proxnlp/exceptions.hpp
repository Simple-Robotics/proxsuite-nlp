/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <stdexcept>
#include <fmt/core.h>

#define PROXNLP_RUNTIME_ERROR(msg)                                             \
  throw std::runtime_error(fmt::format("{}({}): {}", __FILE__, __LINE__, msg))

#define proxnlp_dim_check(x, nx)                                               \
  if (x.size() != nx)                                                          \
  PROXNLP_RUNTIME_ERROR(fmt::format(                                           \
      "Input size invalid (expected {:d}, got {:d})", nx, x.size()))
