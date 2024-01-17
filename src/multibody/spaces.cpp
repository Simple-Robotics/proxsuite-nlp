#include "proxsuite-nlp/modelling/spaces/multibody.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template struct MultibodyConfiguration<context::Scalar>;
template struct MultibodyPhaseSpace<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
