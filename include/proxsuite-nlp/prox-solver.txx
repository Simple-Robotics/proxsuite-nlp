#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/prox-solver.hpp"

namespace proxnlp {

extern template class ProxNLPSolverTpl<context::Scalar>;

} // namespace proxnlp
