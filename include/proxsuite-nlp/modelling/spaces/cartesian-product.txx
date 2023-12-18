#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/spaces/cartesian-product.hpp"

namespace proxnlp {

extern template struct CartesianProductTpl<context::Scalar>;

} // namespace proxnlp
