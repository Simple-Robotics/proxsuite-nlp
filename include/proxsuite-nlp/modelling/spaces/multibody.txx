#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/spaces/multibody.hpp"

namespace proxsuite {
namespace nlp {

extern template struct MultibodyConfiguration<context::Scalar>;
extern template struct MultibodyPhaseSpace<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
