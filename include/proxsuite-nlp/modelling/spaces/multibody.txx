#pragma once

#include "proxsuite-nlp/context.hpp"
#include "proxsuite-nlp/modelling/spaces/multibody.hpp"

namespace proxsuite {
namespace nlp {

extern template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    MultibodyConfiguration<context::Scalar>;
extern template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    MultibodyPhaseSpace<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
