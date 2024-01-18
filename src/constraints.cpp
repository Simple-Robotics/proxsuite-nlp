#include "proxsuite-nlp/modelling/constraints.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ConstraintSetBase<context::Scalar>;
template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ConstraintObjectTpl<context::Scalar>;
template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    EqualityConstraint<context::Scalar>;
template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    NegativeOrthant<context::Scalar>;
template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    BoxConstraintTpl<context::Scalar>;
template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    NonsmoothPenaltyL1Tpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
