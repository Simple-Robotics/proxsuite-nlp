#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/prox-solver.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXTERN template class PROXSUITE_NLP_DLLAPI ProxNLPSolverTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
