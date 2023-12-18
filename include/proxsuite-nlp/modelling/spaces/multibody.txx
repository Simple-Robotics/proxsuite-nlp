#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/spaces/multibody.hpp"

namespace proxnlp {

extern template struct MultibodyConfiguration<context::Scalar>;
extern template struct MultibodyPhaseSpace<context::Scalar>;

} // namespace proxnlp
