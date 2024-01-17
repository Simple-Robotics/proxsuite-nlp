#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/constraints.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI
    ConstraintSetBase<context::Scalar>;
PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI
    ConstraintObjectTpl<context::Scalar>;
PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI
    EqualityConstraint<context::Scalar>;
PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI
    NegativeOrthant<context::Scalar>;
PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI
    BoxConstraintTpl<context::Scalar>;
PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI
    NonsmoothPenaltyL1Tpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
