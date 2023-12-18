#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/spaces/cartesian-product.hpp"

namespace proxsuite {
namespace nlp {

extern template struct CartesianProductTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
