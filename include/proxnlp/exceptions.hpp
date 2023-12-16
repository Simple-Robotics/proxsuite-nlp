/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <stdexcept>
#include <fmt/core.h>

#define PROXNLP_RUNTIME_ERROR(msg)                                             \
  throw ::proxnlp::RuntimeError(                                               \
      fmt::format("{}({}): {}", __FILE__, __LINE__, msg))

#define PROXNLP_DIM_CHECK(x, nx)                                               \
  if (x.size() != nx)                                                          \
  PROXNLP_RUNTIME_ERROR(fmt::format(                                           \
      "Input size invalid (expected {:d}, got {:d})", nx, x.size()))

#define PROXNLP_RAISE_IF_NAN(value)                                            \
  if (::proxnlp::math::check_value(value))                                     \
  PROXNLP_RUNTIME_ERROR("Encountered NaN.\n")

#define PROXNLP_RAISE_IF_NAN_NAME(value, name)                                 \
  if (::proxnlp::math::check_value(value))                                     \
  PROXNLP_RUNTIME_ERROR(fmt::format("Encountered NaN for value {:s}.\n", name))

namespace proxnlp {

class RuntimeError : public std::runtime_error {
public:
  explicit RuntimeError(const std::string &what = "")
      : std::runtime_error(what) {}
};

} // namespace proxnlp
