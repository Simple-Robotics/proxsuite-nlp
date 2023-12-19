#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/costs/quadratic-residual.hpp"

namespace proxsuite {
namespace nlp {

extern template struct QuadraticResidualCostTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
