/// @file linesearch-base.hpp
/// @brief  Base structs for linesearch algorithms.
#pragma once

#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"

#include <fmt/core.h>

namespace proxnlp {
enum LinesearchStrategy { ARMIJO, CUBIC_INTERP };

template <typename Scalar> struct ls_candidate_tpl {
  Scalar alpha;
  Scalar phi;
};

} // namespace proxnlp

#include "proxnlp/linesearch-armijo.hpp"
#include "proxnlp/linesearch-cubic-interp.hpp"
