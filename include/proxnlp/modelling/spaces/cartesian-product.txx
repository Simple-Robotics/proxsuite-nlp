#pragma once

#include "proxnlp/context.hpp"
#include "proxnlp/modelling/spaces/cartesian-product.hpp"

namespace proxnlp {

extern template struct CartesianProductTpl<context::Scalar>;

}
