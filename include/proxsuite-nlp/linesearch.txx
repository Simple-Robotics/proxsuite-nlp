#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/linesearch-base.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION class PROXSUITE_NLP_DLLAPI Linesearch<context::Scalar>;
PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI PolynomialTpl<context::Scalar>;
PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION class PROXSUITE_NLP_DLLAPI ArmijoLinesearch<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
