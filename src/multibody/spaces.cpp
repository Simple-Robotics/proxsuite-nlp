#include "proxsuite-nlp/modelling/spaces/multibody.hpp"

namespace proxsuite {
namespace nlp {

template struct MultibodyConfiguration<context::Scalar>;
template struct MultibodyPhaseSpace<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
