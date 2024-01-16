#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/spaces/cartesian-product.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXTERN template struct PROXSUITE_NLP_DLLAPI CartesianProductTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
