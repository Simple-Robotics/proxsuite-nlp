#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/cost-function.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXP_INST_DECL struct PROXSUITE_NLP_DLLAPI CostFunctionBaseTpl<context::Scalar>;

PROXSUITE_NLP_EXP_INST_DECL PROXSUITE_NLP_DLLAPI auto downcast_function_to_cost<context::Scalar>(
    const shared_ptr<context::C2Function> &) -> shared_ptr<context::Cost>;

} // namespace nlp
} // namespace proxsuite
