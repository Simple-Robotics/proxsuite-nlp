#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/function-ops.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI
    ComposeFunctionTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
