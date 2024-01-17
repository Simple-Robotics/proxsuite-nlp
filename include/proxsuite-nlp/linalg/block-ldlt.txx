#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/linalg/block-ldlt.hpp"

namespace proxsuite {
namespace nlp {
namespace linalg {

PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION struct PROXSUITE_NLP_DLLAPI BlockLDLT<context::Scalar>;

} // namespace linalg
} // namespace nlp
} // namespace proxsuite
