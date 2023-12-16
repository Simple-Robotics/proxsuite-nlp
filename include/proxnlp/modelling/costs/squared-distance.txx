#pragma once

#include "proxnlp/context.hpp"
#include "proxnlp/modelling/costs/squared-distance.hpp"

namespace proxnlp {

extern template struct QuadraticDistanceCostTpl<context::Scalar>;

}
