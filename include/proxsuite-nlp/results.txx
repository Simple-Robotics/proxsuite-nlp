#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/results.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI
    ResultsTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
