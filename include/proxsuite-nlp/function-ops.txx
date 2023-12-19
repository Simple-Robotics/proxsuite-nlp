#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/function-ops.hpp"

namespace proxsuite {
namespace nlp {

extern template struct ComposeFunctionTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
