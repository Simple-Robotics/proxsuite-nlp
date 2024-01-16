#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/problem-base.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXTERN template struct PROXSUITE_NLP_DLLAPI ProblemTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
