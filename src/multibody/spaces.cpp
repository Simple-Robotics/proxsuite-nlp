#include "proxsuite-nlp/modelling/spaces/multibody.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    MultibodyConfiguration<context::Scalar>;
template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    MultibodyPhaseSpace<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
