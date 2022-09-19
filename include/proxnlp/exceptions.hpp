#pragma once

#include <stdexcept>
#include <fmt/core.h>

#define proxnlp_runtime_error(msg)                                             \
  throw std::runtime_error(fmt::format("{}({}): {}", __FILE__, __LINE__, msg))

#define proxnlp_dim_check(x, nx)                                      \
  if (x.size() != nx)                                                          \
  proxnlp_runtime_error(fmt::format(                                           \
      "Input size invalid (expected {:d}, got {:d})", nx, x.size()))
