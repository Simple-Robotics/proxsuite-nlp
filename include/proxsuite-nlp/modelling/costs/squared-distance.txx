#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/costs/squared-distance.hpp"

namespace proxsuite {
namespace nlp {

extern template struct QuadraticDistanceCostTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
