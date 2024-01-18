#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/prox-solver.hpp"

namespace proxsuite {
namespace nlp {

extern template class PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    ProxNLPSolverTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
