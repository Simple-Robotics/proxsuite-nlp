#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/linesearch-base.hpp"

namespace proxsuite {
namespace nlp {

extern template class PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    Linesearch<context::Scalar>;
extern template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    PolynomialTpl<context::Scalar>;
extern template class PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    ArmijoLinesearch<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
