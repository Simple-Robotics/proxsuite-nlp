#pragma once

#include "proxsuite-nlp/context.hpp"
#include "./block-ldlt.hpp"

namespace proxsuite {
namespace nlp {
namespace linalg {

extern template struct BlockLDLT<context::Scalar>;

} // namespace linalg
} // namespace nlp
} // namespace proxsuite
