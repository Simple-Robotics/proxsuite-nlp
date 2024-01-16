#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/function-base.hpp"

namespace proxsuite {
namespace nlp {

extern template struct PROXSUITE_NLP_DLLAPI BaseFunctionTpl<context::Scalar>;

extern template struct PROXSUITE_NLP_DLLAPI C1FunctionTpl<context::Scalar>;

extern template struct PROXSUITE_NLP_DLLAPI C2FunctionTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
