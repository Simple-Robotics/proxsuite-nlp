#include "proxsuite-nlp/modelling/costs/quadratic-residual.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    QuadraticResidualCostTpl<context::Scalar>;

}
} // namespace proxsuite
