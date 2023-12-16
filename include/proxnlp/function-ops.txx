#pragma once

#include "proxnlp/context.hpp"
#include "proxnlp/function-ops.hpp"

namespace proxnlp {

extern template struct ComposeFunctionTpl<context::Scalar>;

} // namespace proxnlp
