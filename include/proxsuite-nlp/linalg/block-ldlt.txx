#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/linalg/block-ldlt.hpp"

namespace proxsuite {
namespace nlp {
namespace linalg {

PROXSUITE_NLP_EXP_INST_DECL struct PROXSUITE_NLP_DLLAPI BlockLDLT<context::Scalar>;

} // namespace linalg
} // namespace nlp
} // namespace proxsuite
