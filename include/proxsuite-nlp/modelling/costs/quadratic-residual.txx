#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/costs/quadratic-residual.hpp"

namespace proxsuite {
namespace nlp {

extern template struct PROXSUITE_NLP_DLLAPI QuadraticResidualCostTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
