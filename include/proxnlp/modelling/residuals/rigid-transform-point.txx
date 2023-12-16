#pragma once

#include "./rigid-transform-point.hpp"
#include "proxnlp/context.hpp"

namespace proxnlp {

extern template struct RigidTransformationPointActionTpl<context::Scalar>;

} // namespace proxnlp
