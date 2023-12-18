#pragma once

#include "proxnlp/context.hpp"
#include "proxnlp/prox-solver.hpp"

namespace proxnlp {

extern template class ProxNLPSolverTpl<context::Scalar>;

} // namespace proxnlp
