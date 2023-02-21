#pragma once

#include "proxnlp/context.hpp"
#include "proxnlp/modelling/costs/squared-distance.hpp"

namespace proxnlp {

extern template struct QuadraticDistanceCost<context::Scalar>;

}
