#include "proxsuite-nlp/function-base.hpp"
#include "proxsuite-nlp/function-ops.hpp"
#include "proxsuite-nlp/cost-function.hpp"
#include "proxsuite-nlp/manifold-base.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    BaseFunctionTpl<context::Scalar>;

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    C1FunctionTpl<context::Scalar>;

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    C2FunctionTpl<context::Scalar>;

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ComposeFunctionTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
