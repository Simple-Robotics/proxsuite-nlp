#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/function-ops.hpp"

namespace proxsuite {
namespace nlp {

extern template struct PROXSUITE_NLP_DLLAPI ComposeFunctionTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
