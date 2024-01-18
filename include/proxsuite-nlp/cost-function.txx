#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/cost-function.hpp"

namespace proxsuite {
namespace nlp {

extern template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    CostFunctionBaseTpl<context::Scalar>;

extern template PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI auto
downcast_function_to_cost<context::Scalar>(
    const shared_ptr<context::C2Function> &) -> shared_ptr<context::Cost>;

} // namespace nlp
} // namespace proxsuite
