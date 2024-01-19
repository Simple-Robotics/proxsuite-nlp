#include "proxsuite-nlp/modelling/costs/squared-distance.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    QuadraticDistanceCostTpl<context::Scalar>;

}
} // namespace proxsuite
