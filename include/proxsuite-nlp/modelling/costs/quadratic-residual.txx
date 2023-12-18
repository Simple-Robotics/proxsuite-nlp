#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/costs/quadratic-residual.hpp"

namespace proxnlp {

extern template struct QuadraticResidualCostTpl<context::Scalar>;

} // namespace proxnlp
