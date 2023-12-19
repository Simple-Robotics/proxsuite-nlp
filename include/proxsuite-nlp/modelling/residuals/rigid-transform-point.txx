#pragma once

#include "./rigid-transform-point.hpp"
#include "proxsuite-nlp/context.hpp"

namespace proxsuite {
namespace nlp {

extern template struct RigidTransformationPointActionTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
