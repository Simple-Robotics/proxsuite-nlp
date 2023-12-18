#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/costs/squared-distance.hpp"

namespace proxnlp {

extern template struct QuadraticDistanceCostTpl<context::Scalar>;

}
