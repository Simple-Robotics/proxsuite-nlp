#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/costs/quadratic-residual.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXP_INST_DECL struct PROXSUITE_NLP_DLLAPI QuadraticResidualCostTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
