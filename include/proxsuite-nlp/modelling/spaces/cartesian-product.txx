#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/spaces/cartesian-product.hpp"

namespace proxsuite {
namespace nlp {

extern template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    CartesianProductTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
