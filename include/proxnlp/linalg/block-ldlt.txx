#pragma once

#include "proxnlp/context.hpp"
#include "./block-ldlt.hpp"

namespace proxnlp {
namespace linalg {

extern template struct BlockLDLT<context::Scalar>;

} // namespace linalg
} // namespace proxnlp
