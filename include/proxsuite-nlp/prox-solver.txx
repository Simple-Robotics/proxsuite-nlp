#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/prox-solver.hpp"

namespace proxsuite {
namespace nlp {

extern template class ProxNLPSolverTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
