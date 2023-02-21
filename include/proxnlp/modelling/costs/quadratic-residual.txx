#pragma once

#include "proxnlp/context.hpp"
#include "proxnlp/modelling/costs/quadratic-residual.hpp"

namespace proxnlp {

extern template struct QuadraticResidualCost<context::Scalar>;

}
