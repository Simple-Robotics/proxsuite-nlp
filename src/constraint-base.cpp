#include "proxsuite-nlp/constraint-base.hpp"

namespace proxsuite::nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ConstraintSetBase<context::Scalar>;
template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ConstraintObjectTpl<context::Scalar>;

} // namespace proxsuite::nlp
