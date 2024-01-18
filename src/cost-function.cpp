#include "proxsuite-nlp/cost-function.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    CostFunctionBaseTpl<context::Scalar>;

template PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI auto
downcast_function_to_cost<context::Scalar>(
    const shared_ptr<context::C2Function> &) -> shared_ptr<context::Cost>;

} // namespace nlp
} // namespace proxsuite
