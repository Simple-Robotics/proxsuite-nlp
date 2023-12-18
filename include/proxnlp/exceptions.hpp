/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <stdexcept>
#include <fmt/core.h>

#define PROXSUITE_NLP_RUNTIME_ERROR(msg)                                       \
  throw ::proxnlp::RuntimeError(                                               \
      fmt::format("{}({}): {}", __FILE__, __LINE__, msg))

#define PROXSUITE_NLP_DIM_CHECK(x, nx)                                         \
  if (x.size() != nx)                                                          \
  PROXSUITE_NLP_RUNTIME_ERROR(fmt::format(                                     \
      "Input size invalid (expected {:d}, got {:d})", nx, x.size()))

#define PROXSUITE_NLP_RAISE_IF_NAN(value)                                      \
  if (::proxnlp::math::check_value(value))                                     \
  PROXSUITE_NLP_RUNTIME_ERROR("Encountered NaN.\n")

#define PROXSUITE_NLP_RAISE_IF_NAN_NAME(value, name)                           \
  if (::proxnlp::math::check_value(value))                                     \
  PROXSUITE_NLP_RUNTIME_ERROR(                                                 \
      fmt::format("Encountered NaN for value {:s}.\n", name))

namespace proxnlp {

class RuntimeError : public std::runtime_error {
public:
  explicit RuntimeError(const std::string &what = "")
      : std::runtime_error(what) {}
};

} // namespace proxnlp
