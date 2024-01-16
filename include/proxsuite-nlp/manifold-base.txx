#pragma once

#include "proxsuite-nlp/config.hpp"
#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/manifold-base.hpp"

namespace proxsuite {
namespace nlp {

PROXSUITE_NLP_EXTERN template struct PROXSUITE_NLP_DLLAPI ManifoldAbstractTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
