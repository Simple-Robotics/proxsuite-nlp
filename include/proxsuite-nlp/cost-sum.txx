#pragma once

#include "./cost-sum.hpp"
#include "proxsuite-nlp/context.hpp"

namespace proxsuite {
namespace nlp {

extern template struct CostSumTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
