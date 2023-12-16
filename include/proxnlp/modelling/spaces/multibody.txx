#pragma once

#include "proxnlp/context.hpp"
#include "proxnlp/modelling/spaces/multibody.hpp"

namespace proxnlp {

extern template struct MultibodyConfiguration<context::Scalar>;
extern template struct MultibodyPhaseSpace<context::Scalar>;

} // namespace proxnlp
